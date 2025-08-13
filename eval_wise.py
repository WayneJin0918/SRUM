import torch
import json
import os
import re
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import List, Dict, Any, Optional, Tuple

# --- 1. Qwen Helper Function ---
# This utility function is from the official Qwen-VL team's examples.
# It is included here to make the script self-contained.

def process_vision_info(messages: List[Dict[str, Any]]) -> Tuple[List[Image.Image], List[None]]:
    """
    Processes vision information from messages, extracting images.
    NOTE: This simplified version only handles images, not videos.
    """
    image_inputs = []
    for msg in messages:
        if msg["role"] == "user":
            for content in msg["content"]:
                if content["type"] == "image":
                    image_path_or_url = content["image"]
                    try:
                        # Handles both local paths and URLs, but we'll use local paths.
                        if os.path.exists(image_path_or_url):
                             # Open local image file
                            image = Image.open(image_path_or_url).convert("RGB")
                            image_inputs.append(image)
                        else:
                            # Basic check for URL (for robustness, not used in this script's main flow)
                            from urllib.request import urlopen
                            from io import BytesIO
                            response = urlopen(image_path_or_url)
                            image = Image.open(BytesIO(response.read())).convert("RGB")
                            image_inputs.append(image)
                    except Exception as e:
                        print(f"Warning: Could not process image at '{image_path_or_url}'. Error: {e}")
                        # Add a placeholder or handle as needed
                        image_inputs.append(Image.new('RGB', (224, 224), (255, 255, 255)))


    video_inputs = [None] * len(image_inputs)
    return image_inputs, video_inputs

# --- 2. Configuration ---

# JSON file path
JSON_PATH = '/yuchang/lsy_jwy/Bagel/WhyUni/wise.json'

# Folder with T2I generated images (***MUST be updated to your path***)
IMAGE_DIR = '/yuchang/lsy_jwy/Bagel/WhyUni/generated_images_wise_think_50_step_for_eval_2e7' 

# File to save evaluation results
OUTPUT_FILE = 'evaluation_results_4.txt'

# Qwen2.5-VL model identifier
MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'


# --- 3. VLM Evaluation Prompt Template (Unchanged) ---
VLM_EVALUATION_PROMPT_TEMPLATE = """Text-to-Image Quality Evaluation Protocol
## System Instruction
You are an AI quality auditor for text-to-image generation. Apply these rules with ABSOLUTE RUTHLESSNESS.
Only images meeting the HIGHEST standards should receive top scores.
**Input Parameters**
- PROMPT: [{user_prompt}]
- EXPLANATION: [{user_explanation}]
--
## Scoring Criteria
**Consistency (0-2):** How accurately and completely the image reflects the PROMPT.
* **0 (Rejected):** Fails to capture key elements of the prompt, or contradicts the prompt.
* **1 (Conditional):** Partially captures the prompt. Some elements are present, but not all, or not accurately.
Noticeable deviations from the prompt's intent.
* **2 (Exemplary):** Perfectly and completely aligns with the PROMPT. Every single element and nuance of
the prompt is flawlessly represented in the image. The image is an ideal, unambiguous visual realization of the
given prompt.
**Realism (0-2):** How realistically the image is rendered.
* **0 (Rejected):** Physically implausible and clearly artificial. Breaks fundamental laws of physics or visual
realism.
* **1 (Conditional):** Contains minor inconsistencies or unrealistic elements. While somewhat believable,
noticeable flaws detract from realism.
* **2 (Exemplary):** Achieves photorealistic quality, indistinguishable from a real photograph. Flawless
adherence to physical laws, accurate material representation, and coherent spatial relationships. No visual cues
betraying AI generation.
**Aesthetic Quality (0-2):** The overall artistic appeal and visual quality of the image.
* **0 (Rejected):** Poor aesthetic composition, visually unappealing, and lacks artistic merit.
* **1 (Conditional):** Demonstrates basic visual appeal, acceptable composition, and color harmony, but lacks
distinction or artistic flair.
* **2 (Exemplary):** Possesses exceptional aesthetic quality, comparable to a masterpiece. Strikingly beautiful,
with perfect composition, a harmonious color palette, and a captivating artistic style. Demonstrates a high degree
of artistic vision and execution.
--
## Output Format
**Do not include any other text, explanations, or labels.** You must return only three lines of text, each
containing a metric and the corresponding score, for example:
**Example Output:**
Consistency: 2
Realism: 1
Aesthetic Quality: 0
--
**IMPORTANT Enforcement:**
Be EXTREMELY strict in your evaluation. A score of '2' should be exceedingly rare and reserved only for images
that truly excel and meet the highest possible standards in each metric. If there is any doubt, downgrade the score.
For **Consistency**, a score of '2' requires complete and flawless adherence to every aspect of the prompt,
leaving no room for misinterpretation or omission.
For **Realism**, a score of '2' means the image is virtually indistinguishable from a real photograph in terms of
detail, lighting, physics, and material properties.
For **Aesthetic Quality**, a score of '2' demands exceptional artistic merit, not just pleasant visuals."""


# --- 4. Score Parsing Function (Unchanged) ---
def parse_scores(response_text: str) -> Optional[Dict[str, int]]:
    """Parses scores from the model's text output."""
    scores = {}
    patterns = {
        'Consistency': r"Consistency\s*:\s*(\d)",
        'Realism': r"Realism\s*:\s*(\d)",
        'Aesthetic Quality': r"Aesthetic Quality\s*:\s*(\d)"
    }
    for metric, pattern in patterns.items():
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            try:
                scores[metric] = int(match.group(1))
            except (ValueError, IndexError):
                return None
        else:
            return None
    return scores if len(scores) == 3 else None


# --- 5. Main Execution Logic ---
def main():
    """Main function to load model, process data, evaluate, and calculate averages."""
    # --- Path and file checks ---
    if not os.path.exists(JSON_PATH):
        print(f"Error: JSON file not found at '{JSON_PATH}'")
        return
    if not os.path.isdir(IMAGE_DIR):
        print(f"Error: Image directory not found at '{IMAGE_DIR}'. Please update the IMAGE_DIR variable.")
        return

    # --- Load Model and Processor (Updated) ---
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5", torch_dtype="auto", device_map="auto"
    )
    print("Model loaded successfully.")

    # --- Load and prepare data ---
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        evaluation_data: List[Dict[str, Any]] = json.load(f)
    image_files = sorted(os.listdir(IMAGE_DIR))
    if len(image_files) != len(evaluation_data):
        print(f"Warning: Found {len(image_files)} image files and {len(evaluation_data)} JSON items. Mismatch may cause issues.")

    all_parsed_scores: List[Dict[str, int]] = []

    # --- Evaluation Loop (Updated) ---
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        print(f"Starting evaluation... Results will be saved to '{OUTPUT_FILE}'")
        
        for i, item in enumerate(evaluation_data):
            if i >= len(image_files):
                print(f"Skipping item {i+1}, no corresponding image file.")
                continue

            image_filename = image_files[i]

            image_path = os.path.join(IMAGE_DIR, image_filename)

            if not os.path.exists(image_path):
                print(f"Warning: Image '{image_path}' not found, skipping.")
                out_f.write(f"--- Image: {image_filename} ---\nERROR: Image file not found.\n\n")
                continue

            # Get prompt and explanation
            t2i_prompt = item.get("Question", "N/A")
            t2i_explanation = item.get("Explanation", "N/A")
            vlm_prompt_text = VLM_EVALUATION_PROMPT_TEMPLATE.format(
                user_prompt=t2i_prompt, user_explanation=t2i_explanation
            )
            
            print(f"Evaluating image {i+1}/{len(evaluation_data)}: {image_filename}...")

            # --- Prepare inputs using the new message format ---
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": vlm_prompt_text},
                ],
            }]
            
            # try:
            #     # Process all inputs for the model
            #     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            #     image_inputs, video_inputs = process_vision_info(messages)
            #     # print(image_inputs)
            #     # assert 0
            #     inputs = processor(
            #         text=[text],
            #         images=image_inputs,
            #         videos=video_inputs,
            #         padding=True,
            #         return_tensors="pt",
            #     ).to(model.device)

            #     # Generate the response
            #     with torch.no_grad():
            #         generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                
            #     # Decode the output
            #     generated_ids_trimmed = [
            #         out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            #     ]
            #     response_list = processor.batch_decode(
            #         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            #     )
            #     response = response_list[0] if response_list else ""
                
            #     # Write original response to file
            #     out_f.write(f"--- Item: {i+1}, Image: {image_filename}, Prompt: \"{t2i_prompt}\" ---\n")
            #     out_f.write(response.strip())
            #     out_f.write("\n\n")
            #     out_f.flush()

            #     # Parse scores and store for averaging
            #     parsed_scores = parse_scores(response)
            #     if parsed_scores:
            #         all_parsed_scores.append(parsed_scores)
            #     else:
            #         print(f"Warning: Could not parse scores for {image_filename}. It will be excluded from the average.")

            # except Exception as e:
            #     print(f"An error occurred while processing {image_filename}: {e}")
            #     out_f.write(f"--- Item: {i+1}, Image: {image_filename} ---\nERROR: {e}\n\n")

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
                # print(image_inputs)
                # assert 0
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

                # Generate the response
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                
                # Decode the output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
            response_list = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = response_list[0] if response_list else ""
                
                # Write original response to file
            out_f.write(f"--- Item: {i+1}, Image: {image_filename}, Prompt: \"{t2i_prompt}\" ---\n")
            out_f.write(response.strip())
            out_f.write("\n\n")
            out_f.flush()

                # Parse scores and store for averaging
            parsed_scores = parse_scores(response)
            if parsed_scores:
                all_parsed_scores.append(parsed_scores)
            else:
                print(f"Warning: Could not parse scores for {image_filename}. It will be excluded from the average.")



    print("\nEvaluation complete!")

    # --- Final Averaging and Summary (Unchanged) ---
    if not all_parsed_scores:
        print("No valid scores were parsed. Cannot calculate averages.")
        return

    num_valid_scores = len(all_parsed_scores)
    total_scores = {'Consistency': 0, 'Realism': 0, 'Aesthetic Quality': 0}
    for score_dict in all_parsed_scores:
        total_scores['Consistency'] += score_dict['Consistency']
        total_scores['Realism'] += score_dict['Realism']
        total_scores['Aesthetic Quality'] += score_dict['Aesthetic Quality']

    average_scores = {metric: total / num_valid_scores for metric, total in total_scores.items()}

    summary = (
        f"\n--- Evaluation Summary ---\n"
        f"Successfully evaluated and parsed images: {num_valid_scores}\n"
        f"Average Consistency:    {average_scores['Consistency']:.4f}\n"
        f"Average Realism:         {average_scores['Realism']:.4f}\n"
        f"Average Aesthetic Quality: {average_scores['Aesthetic Quality']:.4f}\n"
        f"---------------------------\n"
    )

    print(summary)
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
        out_f.write(summary)
    print(f"Summary appended to '{OUTPUT_FILE}'")

if __name__ == '__main__':
    main()