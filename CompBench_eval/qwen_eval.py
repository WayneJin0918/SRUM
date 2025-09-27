# SPDX-License-Identifier: Apache-2.0

import os
import re
import json
import argparse
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
import sys
import logging

import torch
import deepspeed
import torch.distributed as dist
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ==============================================================================
# Helper Functions
# ==============================================================================

def setup_logging(log_file: str, is_main_process: bool):
    """
    Sets up logging. Only the main process writes to a file.
    All processes write to stdout.
    """
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Define handlers: stdout for all, file for main process
    handlers = [logging.StreamHandler(sys.stdout)]
    if is_main_process:
        handlers.append(logging.FileHandler(log_file, mode='w', encoding='utf-8'))
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [RANK %(local_rank)s] - %(message)s',
        handlers=handlers
    )

def load_prompts_from_jsonl(file_path: str) -> Dict[str, str]:
    """Loads prompts from a JSONL file into a dictionary."""
    prompts_map = {}
    logging.info(f"Attempting to load prompts from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'image_file' in data and 'original_prompt' in data:
                        prompts_map[data['image_file']] = data['original_prompt']
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed JSON line: {line.strip()}")
        logging.info(f"Successfully loaded {len(prompts_map)} prompts.")
        return prompts_map
    except FileNotFoundError:
        logging.error(f"Prompt file not found at {file_path}. Cannot proceed.")
        return None

def extract_score(evaluation_text: str) -> int:
    """Extracts a numerical score from the model's output text."""
    try:
        # First attempt: parse as clean JSON
        cleaned_text = evaluation_text.strip().replace('```json', '').replace('```', '').strip()
        match = re.search(r'{\s*.*"score":\s*(\d+).*?}', cleaned_text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if 'score' in data and isinstance(data['score'], int):
                return data['score']
    except (json.JSONDecodeError, TypeError):
        pass  # Fallback to regex if JSON parsing fails
    
    # Second attempt: regex for cases where the score might be a string or JSON is malformed
    pattern = r'"score":\s*"?(\d+)"?'
    match = re.search(pattern, evaluation_text)
    if match:
        return int(match.group(1))
    
    logging.warning(f"Could not extract score from text: '{evaluation_text}'")
    return -1

# ==============================================================================
# Argument Parser
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Use DeepSpeed to evaluate CompBench.")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    return parser.parse_args()

# ==============================================================================
# Core Processing Logic
# ==============================================================================

def process_batch(batch_data: List[Dict], model, processor, device) -> List[Dict]:
    """Processes a single batch of data for inference."""
    batch_results = []
    messages_batch = [[{"role": "user", "content": [{"type": "image", "image": item['image_path']}, {"type": "text", "text": item['question_text']}]}] for item in batch_data]
    
    try:
        text_batch = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
        image_inputs, _ = process_vision_info(messages_batch)
        
        inputs = processor(text=text_batch, images=image_inputs, padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        
        # Decode only the newly generated tokens
        output_texts = processor.batch_decode([out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)], skip_special_tokens=True)
        
        for i, raw_text in enumerate(output_texts):
            result_item = batch_data[i].copy()
            result_item['score'] = extract_score(raw_text)
            result_item['full_evaluation_text'] = raw_text.strip()
            batch_results.append(result_item)
            
    except Exception as e:
        logging.exception("Failed to process a batch.")
        # Create error entries for all items in the failed batch
        for item in batch_data:
            result_item = item.copy()
            result_item['score'] = -1
            result_item['full_evaluation_text'] = f"BATCH_PROCESSING_ERROR: {e}"
            batch_results.append(result_item)
            
    return batch_results

# ==============================================================================
# Main Function
# ==============================================================================

def main():
    args = parse_args()
    
    # Initialize the distributed environment for inter-process communication
    deepspeed.init_distributed()
    
    # Get process information from the distributed environment
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main_process = (rank == 0)
    device = torch.device(f"cuda:{args.local_rank}")

    # --- Logging Setup ---
    # Add local_rank to log records for better debugging
    old_factory = logging.getLogRecordFactory()
    def record_factory(*factory_args, **factory_kwargs):
        record = old_factory(*factory_args, **factory_kwargs)
        record.local_rank = args.local_rank 
        return record
    logging.setLogRecordFactory(record_factory)
    setup_logging(args.log_file, is_main_process)
    
    if is_main_process:
        logging.info(f"Evaluation script started with DeepSpeed on {world_size} GPUs.")
        logging.info(f"Arguments: {vars(args)}")

    # --- Load Prompts ---
    jsonl_path = os.path.join(args.image_path.replace("_no_think", "_think"), "output.jsonl")
    prompts_map = load_prompts_from_jsonl(jsonl_path)
    if prompts_map is None: return

    # --- Load Model with DeepSpeed (includes synchronized loading to avoid resource contention) ---
    logging.info(f"Loading model: {args.model_id}. This may take a while...")
    try:
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        
        # Synchronize model loading to avoid resource contention
        # Step 1: All non-main processes wait here.
        if not is_main_process:
            dist.barrier()

        # Step 2: The main process (rank 0) loads the model from disk first.
        # This populates the OS file cache, making subsequent process loads much faster.
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        # Step 3: After the main process is done, it signals other processes to start loading (which will now hit the cache).
        if is_main_process:
            dist.barrier()
        
        # Now, all processes have the model on CPU, ready for DeepSpeed to partition and move to respective GPUs.
        model = deepspeed.init_inference(
            model,
            dtype=torch.bfloat16,
            replace_with_kernel_inject=False, # Set to False for VL models or if custom kernels cause issues
            max_tokens=4096
        )
        logging.info(f"Rank {rank} successfully initialized model with DeepSpeed.")
    except Exception:
        logging.exception("Failed to load or initialize model.")
        return

    # --- Prepare Evaluation Tasks (all processes do this, it's fast) ---
    logging.info("Preparing evaluation tasks...")
    final_tasks = []
    image_path_total = os.path.join(args.image_path, "samples")
    if not os.path.exists(image_path_total): image_path_total = args.image_path
    if not os.path.isdir(image_path_total):
        logging.error(f"Image directory not found: {image_path_total}")
        return
        
    image_files = sorted(
        [f for f in os.listdir(image_path_total) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(os.path.splitext(x)[0]) # Sort numerically by filename
    )
    
    question_text_template = (
        "You are my assistant to evaluate the correspondence of the image to a given text prompt. "
        "Focus on the objects in the image and their attributes (such as color, shape, texture), spatial layout and action relationships. "
        "According to the image, evaluate how well the image aligns with the text prompt: \"{prompt_name}\" "
        "Give a score from 0 to 100, according the criteria: \n"
        "81-100: the image perfectly matches the content of the text prompt, with no discrepancies. \n"
        "61-80: the image portrayed most of the actions, events and relationships but with minor discrepancies. \n"
        "41-60: the image depicted some elements in the text prompt, but ignored some key parts or details. \n"
        "21-40: the image did not depict any actions or events that match the text. \n"
        "1-20: the image failed to convey the full scope in the text prompt. \n"
        "Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2), explanation (within 20 words)."
    )

    for image_file in image_files:
        prompt_name = prompts_map.get(image_file)
        if prompt_name:
            final_tasks.append({
                "question_id": int(os.path.splitext(image_file)[0]),
                "prompt": prompt_name,
                "category": args.category,
                "image_path": os.path.join(image_path_total, image_file),
                "question_text": question_text_template.format(prompt_name=prompt_name)
            })

    if not final_tasks:
        logging.warning("No tasks to process after applying filters.")
        return

    # --- Batching and Data Sharding ---
    all_batches = [final_tasks[i:i + args.batch_size] for i in range(0, len(final_tasks), args.batch_size)]
    
    # Each process gets only its own subset of data to process (division of labor).
    batches_for_this_rank = all_batches[rank::world_size]
    
    if is_main_process:
        logging.info(f"Total {len(all_batches)} batches split across {world_size} GPUs.")
        if os.path.exists(args.output_csv):
            logging.info(f"Removing existing output file: {args.output_csv}")
            os.remove(args.output_csv)

    # --- Inference ---
    # Each process only handles its own portion of the batches.
    my_results = []
    progress_bar = tqdm(batches_for_this_rank, desc=f"Rank {rank} Processing", disable=not is_main_process)
    for batch in progress_bar:
        batch_results = process_batch(batch, model, processor, device)
        my_results.extend(batch_results)

    # --- Gather results from all processes (aggregation) ---
    logging.info(f"Rank {rank} finished processing {len(my_results)} results. Waiting to gather...")
    
    # Create a placeholder list to receive results from all processes.
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, my_results)
    
    # --- Save Results (executed only on the main process) ---
    if is_main_process:
        logging.info("All ranks have finished. Main process is saving the results.")
        
        # Flatten the gathered list of results (a list of lists) into a single list.
        final_results = [item for sublist in gathered_results for item in sublist]
        
        logging.info(f"Gathered a total of {len(final_results)} results. Saving...")
        if final_results:
            final_df = pd.DataFrame(final_results)
            # Ensure final output is sorted by question_id for consistency
            if 'question_id' in final_df.columns:
                final_df.sort_values(by='question_id', inplace=True)
            
            final_df.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
            
            # Append a statistical summary to the CSV
            valid_scores = final_df[final_df['score'] >= 0]['score']
            avg_score = valid_scores.mean() if not valid_scores.empty else 0
            stats_header = "\n\n# --- Evaluation Summary --- #"
            stats_body = f"\nTotal Items,{len(final_df)}\nValid Scores,{len(valid_scores)}\nAverage Score,{avg_score:.4f}"
            with open(args.output_csv, 'a', encoding='utf-8-sig', newline='') as f:
                f.write(stats_header)
                f.write(stats_body)
            
            logging.info(f"Total Items Evaluated: {len(final_df)}")
            logging.info(f"Average Score for this run: {avg_score:.4f}")
        else:
            logging.warning("No results were generated to save.")

    logging.info(f"Rank {rank} finished script successfully.")

if __name__ == "__main__":
    main()