import os
import json
import argparse
import torch
import multiprocessing
from pathlib import Path

def run_on_gpu(proc_id, gpu_id, prompts, args, tmp_jsonl_path):
    """
    A target function to run the inference script on a specific GPU.
    This function is executed by a single process.
    """
    # Set the current process to use the assigned GPU card
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Dynamically import the main inference script and its dependencies
    # This is done here to ensure it happens within the new process
    from srum_data_infer.t2i import main as t2i_main
    from srum_data_infer.t2i import parser as t2i_parser # Import the parser from the inference script

    # Create a temporary metadata file unique to this process
    # Using proc_id ensures no file conflicts between processes on the same GPU
    tmp_metadata_path = f"tmp_metadata_proc_{proc_id}.json"
    with open(tmp_metadata_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    # Construct arguments for the inference script
    sub_args_list = [
        '--model_path', args.model_path,
        '--max_mem_per_gpu', args.max_mem_per_gpu,
        '--output_dir', args.output_dir,
        '--metadata_file', tmp_metadata_path,
        '--num_timesteps', args.num_timesteps,
        '--jsonl_output_path', str(tmp_jsonl_path),
    ]
    if args.think:
        sub_args_list.append('--think')
    if args.overwrite:
        sub_args_list.append('--overwrite')

    sub_args = t2i_parser.parse_args(sub_args_list)

    print(f"[GPU {gpu_id} | Process {proc_id}] Starting inference with {len(prompts)} prompts.")
    # Call the main inference function
    t2i_main(sub_args)
    print(f"[GPU {gpu_id} | Process {proc_id}] Finished inference.")

    # Clean up the temporary metadata file
    os.remove(tmp_metadata_path)

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU launcher for BAGEL model inference.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_mem_per_gpu", type=str, default="80GiB")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--metadata_file", type=str, required=True)
    parser.add_argument("--think", action="store_true", help="Enable 'think' mode to generate detailed prompts.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing image files.")
    parser.add_argument("--num_timesteps", type=str, default="100")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    
    # NEW: Add argument to define number of processes per GPU
    parser.add_argument("--procs_per_gpu", type=int, default=2, help="Number of processes to launch per GPU.")
    
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read all prompts and pre-assign IDs
    print("Preparing and pre-assigning IDs to prompts...")
    with open(args.metadata_file, "r", encoding="utf-8") as f:
        metadatas = json.load(f)

    for i, meta in enumerate(metadatas):
        meta['prompt_id'] = meta.get('prompt_id', f"image_{i:05d}")
        meta['original_index'] = i

    # 2. Distribute prompts across the total number of processes
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    if num_gpus == 0:
        raise ConnectionError("No GPUs detected. This script requires at least one GPU.")
    
    # MODIFIED: Calculate total processes and distribute work
    total_processes = num_gpus * args.procs_per_gpu
    if total_processes == 0:
        raise ValueError("Total number of processes is zero. Check --num_gpus and --procs_per_gpu.")
        
    chunks = [metadatas[i::total_processes] for i in range(total_processes)]
    print(f"Distributing {len(metadatas)} prompts across {num_gpus} GPUs with {args.procs_per_gpu} processes per GPU ({total_processes} total processes).")

    # 3. Launch parallel processes
    processes = []
    tmp_jsonl_paths = []
    # MODIFIED: Loop through total processes and assign each to a GPU
    for proc_id, prompts_chunk in enumerate(chunks):
        if not prompts_chunk:
            continue
            
        # Assign process to a GPU in a round-robin fashion
        gpu_id = proc_id % num_gpus
        
        # Define a unique temporary file for each process to write its results
        tmp_jsonl_path = output_dir / f"tmp_results_proc_{proc_id}.jsonl"
        tmp_jsonl_paths.append(tmp_jsonl_path)
        
        # Pass both proc_id and gpu_id to the target function
        p = multiprocessing.Process(target=run_on_gpu, args=(proc_id, gpu_id, prompts_chunk, args, tmp_jsonl_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All inference processes have completed.")

    # 4. Consolidate results (no changes needed here)
    if args.think:
        print("Consolidating and sorting results...")
        all_records = []
        for path in tmp_jsonl_paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        all_records.append(json.loads(line))
                os.remove(path) # Clean up temporary file

        all_records.sort(key=lambda x: x['original_index'])

        final_jsonl_path = output_dir / "output.jsonl"
        with open(final_jsonl_path, "w", encoding="utf-8") as f:
            for record in all_records:
                del record['original_index']
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Successfully created consolidated results file at: {final_jsonl_path}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()