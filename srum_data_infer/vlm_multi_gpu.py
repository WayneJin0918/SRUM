# Filename: vlm_multi_gpu.py

import os
import json
import argparse
import torch
import multiprocessing
from pathlib import Path
import copy
import logging

def setup_logging(log_file):
    """Set up a logger to output to both a file and the console"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        return

    formatter = logging.Formatter(
        '%(asctime)s - %(processName)s (%(process)d) - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def run_analysis_on_gpu(process_id, gpu_id, records_chunk, args):
    """
    Target function to run the core logic of vlm_analysis.py on a specified GPU.
    """
    if args.log_file:
        setup_logging(args.log_file)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        from vlm import main as vlm_main
    except ImportError:
        logging.error(f"Error: Failed to import 'vlm.py'. Make sure it's in the same directory or Python path.")
        return

    output_dir = Path(os.path.dirname(args.output_jsonl))
    tmp_input_jsonl = output_dir / f"tmp_input_proc_{process_id}.jsonl"
    tmp_output_jsonl = output_dir / f"tmp_rewards_proc_{process_id}.jsonl"

    with open(tmp_input_jsonl, "w", encoding="utf-8") as f:
        for record in records_chunk:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    sub_args = copy.deepcopy(args)
    sub_args.input_jsonl = str(tmp_input_jsonl)
    sub_args.output_jsonl = str(tmp_output_jsonl)

    logging.info(f"Process-{process_id} starting analysis with {len(records_chunk)} images on GPU {gpu_id}.")
    try:
        vlm_main(sub_args)
        logging.info(f"Process-{process_id} finished VLM analysis on GPU {gpu_id}.")
    except Exception as e:
        logging.error(f"An error occurred during execution in Process-{process_id} on GPU {gpu_id}: {e}", exc_info=True)
    finally:
        if os.path.exists(tmp_input_jsonl):
            os.remove(tmp_input_jsonl)


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU launcher for VLM analysis.")
    
    # --- Launcher-specific arguments ---
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use.")
    parser.add_argument("--processes_per_gpu", type=int, default=1, help="Number of parallel processes to run on each GPU.")
    parser.add_argument("--log_file", type=str, default="vlm_multi_gpu.log", help="Path to the log file.")

    # --- Arguments for the child script (vlm.py) ---
    parser.add_argument("--model_path", type=str, required=True, help="Path to the BAGEL model checkpoint directory.")
    parser.add_argument("--max_mem_per_gpu", type=str, default="80GiB", help="Maximum memory per GPU.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory where generated images are stored.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="The main input JSONL file.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="The final consolidated output file.")
    
    # --- MODIFIED: Arguments synchronized with the new vlm.py ---
    parser.add_argument("--sam_model_path", type=str, default="sam_vit_h_4b8939.pth", help="Path to the SAM model checkpoint file (for region-based mode).")
    parser.add_argument("--sam_model_type", type=str, default="vit_h", help="The type of SAM model (for region-based mode).")
    parser.add_argument("--max_regions", type=int, default=10, help="Max regions to evaluate in region-based mode.")
    parser.add_argument("--max_retries", type=int, default=5, help="Maximum number of retries if VLM output parsing fails.")
    
    # --- MODIFIED: New control flags from vlm.py ---
    parser.add_argument("--sample_level_only", action="store_true", 
                        help="If set, performs a single evaluation on the entire image, ignoring regions.")
    parser.add_argument("--global_layout_reward", action="store_true",
                        help="If set, adds a separate global reward that evaluates only the layout and positional relationships of objects.")
    parser.add_argument("--binarize_score", action="store_true", 
                        help="If set, adds a 'binarized_score' field (1 for score > 0, -1 otherwise).")

    # --- Launcher control ---
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the final consolidated output file.")

    args = parser.parse_args()
    
    setup_logging(args.log_file)

    output_dir = Path(os.path.dirname(args.output_jsonl))
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_ids = set()
    UNIQUE_KEY = 'image_file'  

    # Calculate total potential processes to determine temp filenames
    num_gpus_avail = min(args.num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    total_processes = num_gpus_avail * args.processes_per_gpu

    if args.overwrite:
        logging.warning(f"Overwrite flag is set. Removing existing output and temporary files.")
        if os.path.exists(args.output_jsonl):
            os.remove(args.output_jsonl)
        for i in range(total_processes):
            tmp_file = output_dir / f"tmp_rewards_proc_{i}.jsonl"
            if tmp_file.exists():
                os.remove(tmp_file)
    else:
        # --- MODIFIED: More Robust Resume Logic ---
        # 1. Check final output file for completed work
        if os.path.exists(args.output_jsonl):
            logging.info(f"Checking final output for completed records: {args.output_jsonl}")
            try:
                with open(args.output_jsonl, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            if UNIQUE_KEY in record:
                                completed_ids.add(record[UNIQUE_KEY])
                        except json.JSONDecodeError:
                            continue # Skip malformed lines
                logging.info(f"Found {len(completed_ids)} records in the final output file.")
            except Exception as e:
                logging.error(f"Could not read existing output file. Please fix or use --overwrite. Error: {e}")
                return

        # 2. ALSO check temporary files for any work from a previous crashed run
        temp_ids_found = 0
        for i in range(total_processes):
            tmp_file = output_dir / f"tmp_rewards_proc_{i}.jsonl"
            if tmp_file.exists():
                try:
                    with open(tmp_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                record = json.loads(line)
                                if UNIQUE_KEY in record and record[UNIQUE_KEY] not in completed_ids:
                                    completed_ids.add(record[UNIQUE_KEY])
                                    temp_ids_found += 1
                            except json.JSONDecodeError:
                                continue # Skip malformed lines
                except Exception as e:
                    logging.warning(f"Could not read temporary file {tmp_file}. It might be corrupted. Skipping. Error: {e}")
        if temp_ids_found > 0:
            logging.info(f"Found {temp_ids_found} additional records in leftover temporary files.")
        
        if completed_ids:
             logging.info(f"Total unique completed records found: {len(completed_ids)}. These will be skipped.")

    logging.info(f"Reading input metadata from {args.input_jsonl}...")
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f]

    # Filter out records that have already been processed
    if completed_ids:
        original_count = len(all_records)
        records_to_process = [r for r in all_records if r.get(UNIQUE_KEY) not in completed_ids]
        logging.info(f"Filtered records: {len(records_to_process)} to process out of {original_count} total.")
    else:
        records_to_process = all_records
    
    if not records_to_process:
        logging.info("All records have already been processed. Consolidating any remaining temp files and exiting.")
        # Fall through to the consolidation step to clean up any temp files, then exit.
    else:
        # --- This part runs only if there's work to do ---
        num_gpus = min(args.num_gpus, torch.cuda.device_count())
        if num_gpus == 0:
            logging.critical("No GPUs detected. Aborting.")
            raise ConnectionError("No GPUs detected.")
        
        # Recalculate total_processes in case GPU count was 0 before
        total_processes = num_gpus * args.processes_per_gpu
        chunks = [records_to_process[i::total_processes] for i in range(total_processes)]
        logging.info(f"Distributing {len(records_to_process)} records across {total_processes} processes on {num_gpus} GPUs.")

        processes = []
        for i in range(total_processes):
            if not chunks[i]:
                continue
            
            gpu_id_for_process = i % num_gpus
            records_chunk = chunks[i]
            
            p = multiprocessing.Process(
                target=run_analysis_on_gpu, 
                args=(i, gpu_id_for_process, records_chunk, args),
                name=f"Process-{i}"
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        logging.info("All VLM analysis processes have completed.")
    
    # --- Final Consolidation (runs always to ensure cleanup) ---
    logging.info(f"Consolidating all results into {args.output_jsonl}...")
    final_write_mode = "a" if not args.overwrite and os.path.exists(args.output_jsonl) else "w"
    
    with open(args.output_jsonl, final_write_mode, encoding="utf-8") as final_f:
        for i in range(total_processes):
            tmp_path = output_dir / f"tmp_rewards_proc_{i}.jsonl"
            if tmp_path.exists():
                with open(tmp_path, "r", encoding="utf-8") as tmp_f:
                    final_f.write(tmp_f.read())
                os.remove(tmp_path) 

    logging.info(f"Successfully created/updated consolidated rewards file at: {args.output_jsonl}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()