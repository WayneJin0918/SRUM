# SPDX-License-Identifier: Apache-2.0

import os
import re
import json
import argparse
import pandas as pd
from typing import List, Dict, Any
import concurrent.futures
from tqdm import tqdm
import sys
import logging

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def setup_logging(log_file: str):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_prompts_from_jsonl(file_path: str) -> Dict[str, str]:
    prompts_map = {}
    logging.info(f"Attempting to load prompts from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'image_file' in data and 'original_prompt' in data:
                        prompts_map[data['image_file']] = data['original_prompt']
                    else:
                        logging.warning(f"Skipping line in JSONL due to missing 'image_file' or 'original_prompt': {line.strip()}")
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed JSON line: {line.strip()}")
        logging.info(f"Successfully loaded {len(prompts_map)} prompts.")
        return prompts_map
    except FileNotFoundError:
        logging.error(f"Prompt file not found at {file_path}. Cannot proceed.")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="Use local Qwen-VL to evaluate CompBench.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the directory containing image samples.")
    parser.add_argument("--category", type=str, required=True, help="Category to evaluate.")
    parser.add_argument("--start", type=int, default=0, help="Start index of images to evaluate.")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive) of images to evaluate.")
    parser.add_argument("--step", type=int, default=1, help="Step for iterating through images (1 means all images).")
    parser.add_argument("--output_csv", type=str, default="./compbench_qwen_results.csv", help="Path to save the output CSV file.")
    parser.add_argument("--log_file", type=str, default="evaluation.log", help="Path to save the log file.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="The Qwen-VL model to use.")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of images to process in a single batch.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of parallel worker threads.")
    return parser.parse_args()

def extract_score(evaluation_text: str) -> int:
    try:
        cleaned_text = evaluation_text.strip().replace('```json', '').replace('```', '').strip()
        match = re.search(r'{\s*.*"score":\s*(\d+).*?}', cleaned_text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if 'score' in data and isinstance(data['score'], int):
                return data['score']
    except (json.JSONDecodeError, TypeError) as e:
        logging.warning(f"Failed to parse score from JSON. Text: '{evaluation_text}'. Error: {e}")
    pattern = r'"score":\s*"?(\d+)"?'
    match = re.search(pattern, evaluation_text)
    if match: return int(match.group(1))
    logging.warning(f"Could not extract score from text: '{evaluation_text}'")
    return -1

def save_to_csv(results: List[Dict], output_path: str):
    if not results: logging.info("No results to save."); return
    df = pd.DataFrame(results)
    required_cols = ['question_id', 'prompt', 'category', 'score', 'full_evaluation_text']
    for col in required_cols:
        if col not in df.columns: df[col] = None
    df.sort_values(by='question_id', inplace=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logging.info(f"Successfully saved {len(df)} results to {output_path}")
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    valid_scores_df = df[df['score'] >= 0]
    total_count, successful_count = len(df), len(valid_scores_df)
    error_count = total_count - successful_count
    avg_score = valid_scores_df['score'].mean() if successful_count > 0 else 0
    stats_header = "\n\n# --- Evaluation Summary --- #"; stats_body = f"\nTotal Items Evaluated,{total_count}\nSuccessfully Scored,{successful_count}\nErrors (Score=-1),{error_count}\n\nMetric,Average Score\nOverall Score,{avg_score:.4f}\n"
    try:
        with open(output_path, 'a', encoding='utf-8-sig', newline='') as f: f.write(stats_header); f.write(stats_body)
        logging.info("Successfully appended summary statistics.")
    except Exception: logging.exception(f"Could not append statistics to CSV: {output_path}")

def process_batch(batch_data: List[Dict], model, processor) -> List[Dict]:
    batch_results, messages_batch = [], []
    for item in batch_data: messages_batch.append([{"role": "user", "content": [{"type": "image", "image": item['image_path']}, {"type": "text", "text": item['question_text']}]}])
    try:
        text_batch = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
        image_inputs, _ = process_vision_info(messages_batch)
        inputs = processor(text=text_batch, images=image_inputs, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad(): generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        output_texts = processor.batch_decode([out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)], skip_special_tokens=True)
        for i, raw_text in enumerate(output_texts):
            result_item = batch_data[i]
            result_item['score'] = extract_score(raw_text)
            result_item['full_evaluation_text'] = raw_text.strip()
            batch_results.append(result_item)
    except Exception as e:
        logging.exception("Failed to process a batch.")
        for item in batch_data: item['score'] = -1; item['full_evaluation_text'] = f"BATCH_PROCESSING_ERROR: {e}"; batch_results.append(item)
    return batch_results

def main():
    args = parse_args()
    setup_logging(args.log_file)
    logging.info("Evaluation script started.")
    logging.info(f"Arguments: {vars(args)}")

    current_image_path = args.image_path.rstrip('/')
    if "_no_think" in os.path.basename(current_image_path):
        think_path = current_image_path.replace("_no_think", "_think")
    else:
        think_path = current_image_path
    jsonl_path = os.path.join(think_path, "output.jsonl")
    
    prompts_map = load_prompts_from_jsonl(jsonl_path)
    if prompts_map is None:
        return

    logging.info(f"Loading model: {args.model_id}. This may take a while...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        logging.info("Model and processor loaded successfully.")
    except Exception:
        logging.exception("Failed to load model.")
        return

    logging.info("Preparing evaluation tasks...")
    tasks_to_process = []
    try:
        image_path_total = os.path.join(current_image_path, "samples")
        if not os.path.exists(image_path_total): image_path_total = current_image_path
        
        ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        all_files_in_dir = os.listdir(image_path_total)
        image_files = [f for f in all_files_in_dir if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS]
        
        if not image_files: logging.error(f"No image files found in {image_path_total}"); return
        logging.info(f"Found {len(image_files)} image files to process.")
        
        image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except Exception:
        logging.exception(f"Error reading or sorting image files from {current_image_path}.")
        return

    for image_file in image_files:
        try:
            prompt_name = prompts_map.get(image_file)
            if prompt_name is None:
                logging.warning(f"No prompt found for image_file '{image_file}'. Skipping.")
                continue

            question_id = int(os.path.splitext(image_file)[0])
            image_path = os.path.join(image_path_total, image_file)
            
            question_text = (
                f"You are my assistant to evaluate the correspondence of the image to a given text prompt. "
                f"Focus on the objects in the image and their attributes (such as color, shape, texture), spatial layout and action relationships. "
                f"According to the image, evaluate how well the image aligns with the text prompt: \"{prompt_name}\" "
                f"Give a score from 0 to 100, according the criteria: \n"
                f"81-100: the image perfectly matches the content of the text prompt, with no discrepancies. \n"
                f"61-80: the image portrayed most of the actions, events and relationships but with minor discrepancies. \n"
                f"41-60: the image depicted some elements in the text prompt, but ignored some key parts or details. \n"
                f"21-40: the image did not depict any actions or events that match the text. \n"
                f"1-20: the image failed to convey the full scope in the text prompt. \n"
                f"Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2), explanation (within 20 words)."
            )
            tasks_to_process.append({"question_id": question_id, "prompt": prompt_name, "category": args.category, "image_path": image_path, "question_text": question_text})
        except ValueError:
            logging.warning(f"Skipping file with non-numeric name: {image_file}"); continue
    
    # [修改] 应用 start, end, step 参数来切分任务
    final_tasks = tasks_to_process[args.start:args.end:args.step]
    if not final_tasks: logging.warning("No tasks to process after applying start/end/step filters."); return
    logging.info(f"Generated {len(final_tasks)} tasks to process for this chunk.")

    batches = [final_tasks[i:i + args.batch_size] for i in range(0, len(final_tasks), args.batch_size)]
    logging.info(f"Data split into {len(batches)} batches of size up to {args.batch_size}.")

    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_batch = {executor.submit(process_batch, batch, model, processor): batch for batch in batches}
        
        progress_bar = tqdm(concurrent.futures.as_completed(future_to_batch), total=len(batches), desc=f"Processing chunk {args.start}-{args.end}")
        for future in progress_bar:
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
            except Exception as exc:
                logging.exception(f"A batch generated an exception: {exc}")

    logging.info("Saving final results...")
    save_to_csv(all_results, args.output_csv)
    logging.info("Evaluation script finished successfully.")

if __name__ == "__main__":
    main()