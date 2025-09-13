# filename: vlm_analysis.py

import os
import re
import json
import argparse
from PIL import Image
import torch
from tqdm import tqdm
import logging
import numpy as np

# --- Dependencies for 'unsupervised' mode ---
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    import cv2
    UNSUPERVISED_DEPS_AVAILABLE = True
except ImportError:
    UNSUPERVISED_DEPS_AVAILABLE = False

# --- Original model dependencies ---
# Note: These are placeholder imports. You must have the actual model files and dependencies installed.
from inferencer import InterleaveInferencer
from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 1. Prompt and Parsing Functions ---

def create_sample_level_evaluation_prompt(original_prompt: str) -> str:
    """
    Generates a VLM prompt for scoring the entire image as a single sample.
    """
    return f"""# TASK: Whole Image Analysis and Scoring
You are an expert AI image analyst. Your task is to analyze the entire image based on a user's prompt and provide a single, overall score.

**Original Prompt:** "{original_prompt}"
---
## YOUR TASK & OUTPUT FORMAT

### STAGE 1: Overall Impression
First, briefly describe your overall impression of the image in relation to the prompt.
* **Output Line:** `Overall Impression: [Your description]`

### STAGE 2: Score and Justify
Provide a single, overall score from **-1.0 to 1.0** that considers how well the entire image satisfies the prompt, including its **relevance** and **visual quality**. Be as strict as possible and only give full marks when the image quality is beyond doubt.

* **Scoring Guide:**
    * **1.0:** Perfect. The image is exactly what the prompt asks for and is technically flawless.
    * **0.5 to 0.9:** Very good. A highly relevant image with minor flaws.
    * **-0.4 to 0.4:** Neutral/Acceptable. A moderately relevant image or an image with mixed qualities.
    * **-0.9 to -0.5:** Bad. The image is largely irrelevant or has severe visual flaws.
    * **-1.0:** Very Bad. The image actively undermines the prompt's intent.

* **Output Lines:**
    `Score: [A single number between -1.0 and 1.0]`
    `Reason: [Your justification for the score]`
---
## EXAMPLE OUTPUT STRUCTURE
Overall Impression: A well-composed photo of a golden retriever running in a field.
Score: 0.95
Reason: The image perfectly captures the prompt's subject with good lighting and focus.
---
Begin your analysis now.
"""

def create_global_layout_reward_prompt(original_prompt: str) -> str:
    """
    Generates a streamlined VLM prompt for evaluating the overall image composition
    and its alignment with the user's prompt, optimized for smaller models.
    """
    return f"""# TASK: Global Layout and Composition Analysis
You are an expert image analyst. Your task is to score the overall composition of an image based on a user's prompt. Focus solely on how the arrangement of elements and scene structure align with the prompt's spatial intent.

**Original Prompt:** "{original_prompt}"
---
## YOUR TASK & OUTPUT FORMAT
Provide a single score from **-1.0 to 1.0** and a brief reason.

* **Scoring Guide:**
    * **1.0:** Perfect alignment with the prompt's spatial intent.
    * **0.5 to 0.9:** Mostly correct layout with minor flaws.
    * **-0.4 to 0.4:** Neutral. No specific spatial info in prompt, or generic layout.
    * **-0.9 to -0.5:** Incorrect layout or contradictory to the prompt.
    * **-1.0:** Fundamentally contradicts the prompt's spatial intent.

* **Output Lines:**
    `Score: [A single number between -1.0 and 1.0]`
    `Reason: [Your justification]`
---
## DIVERSE EXAMPLES

### Example 1 (Perfect Alignment)
Score: 0.95
Reason: The wide shot of a sunset over the ocean perfectly matches the prompt's implied composition.

### Example 2 (Contradictory Layout)
Score: -0.7
Reason: The cat is on the right of the dog, but the prompt asked for the cat on the left.

### Example 3 (Neutral Layout)
Score: 0.2
Reason: The prompt 'a picture of a tree' has no specific compositional requirements, so the layout is neutral.
---
Begin your analysis now.
"""

def parse_sample_level_output(vlm_text_output: str) -> dict:
    """
    Parses the VLM's output for a single, sample-level evaluation.
    """
    evaluation = {}
    content = vlm_text_output.split('</think>')[-1].strip()
    
    try:
        impression_match = re.search(r"Overall Impression:\s*(.*)", content, re.IGNORECASE)
        score_match = re.search(r"Score:\s*(-?\d+\.?\d*)", content, re.IGNORECASE)
        reason_match = re.search(r"Reason:\s*(.*)", content, re.IGNORECASE | re.DOTALL)

        if impression_match:
            evaluation['identified_object'] = impression_match.group(1).strip()
        if score_match:
            evaluation['score'] = float(score_match.group(1))
        if reason_match:
            evaluation['reason'] = reason_match.group(1).strip()
            
    except (ValueError, IndexError, AttributeError) as e:
        logging.warning(f"Failed to parse sample-level output. Error: {e}\nContent:\n{content}")

    return evaluation

def parse_global_layout_output(vlm_text_output: str) -> dict:
    """
    Parses the VLM's output for a single, global layout evaluation.
    This version uses more robust regex to handle various output formats.
    """
    evaluation = {}
    content = vlm_text_output.split('</think>')[-1].strip()
    
    # Use a more robust regex that can handle multi-line outputs and various spacing
    # The '|' in the score regex handles both floats and integers
    score_pattern = r"Score:\s*(-?\d+\.?\d*)"
    layout_pattern = r"Composition Analysis:\s*(.*)"
    reason_pattern = r"Reason:\s*(.*)"
    
    # Use re.DOTALL to allow matching across newlines
    try:
        # Match layout analysis
        layout_match = re.search(layout_pattern, content, re.IGNORECASE | re.DOTALL)
        if layout_match:
            evaluation['layout_analysis'] = layout_match.group(1).strip()
            
        # Match score
        score_match = re.search(score_pattern, content, re.IGNORECASE | re.DOTALL)
        if score_match:
            evaluation['global_layout_score'] = float(score_match.group(1))

        # Match reason
        # We can extract the reason as the rest of the text after the score
        if score_match:
            reason_text = content[score_match.end():].strip()
            if reason_text.lower().startswith('reason:'):
                 evaluation['reason'] = reason_text[len('reason:'):].strip()
            else:
                 evaluation['reason'] = reason_text # Fallback to capture any text after the score
        
    except (ValueError, IndexError, AttributeError) as e:
        # This warning is crucial for debugging
        logging.warning(f"Failed to parse global layout output. Error: {e}\nContent:\n{content}")
    
    # Final sanity check: if score is not found, return empty dict
    if 'global_layout_score' not in evaluation:
        return {}

    return evaluation

def create_hybrid_evaluation_prompt(original_prompt: str, detected_objects: list[dict]) -> str:
    """
    Generates a VLM prompt for scoring multiple regions in an image.
    """
    regions_text = ""
    for i, obj in enumerate(detected_objects):
        region_id = i + 1
        obj['region_id'] = region_id
        regions_text += f"Region ID: {region_id}\nBounding Box: {obj['bbox']}\n---\n"

    return f"""# TASK: Integrated Region Analysis and Scoring
You are an expert AI image analyst. Your task is to analyze unlabeled regions in an image based on a user's prompt. For each region, you will perform a two-stage analysis.

**Original Prompt:** "{original_prompt}"
---
**UNLABELED REGIONS FOR YOUR ANALYSIS:**
{regions_text}
---
## YOUR TWO-STAGE TASK & OUTPUT FORMAT
For **every Region ID** listed above, you must perform the following steps.

### STAGE 1: Identify Object
First, identify the primary object within the bounding box.
* **Output Line:** `Identified Object: [Your description of the object]`

### STAGE 2: Score and Justify
Provide a single, overall score from **-1.0 to 1.0** that considers BOTH the object's **relevance** to the prompt and its **visual quality**. You must provide a clear reason for your score. Be as strict as possible and only give full marks when the image quality is beyond doubt.

* **Scoring Guide:**
    * **1.0:** Perfect. The object is exactly what the prompt asks for and is technically flawless and perfect.
    * **0.5 to 0.9:** Very good. A highly relevant object with minor flaws, or a well-executed secondary element.
    * **-0.4 to 0.4:** Neutral/Acceptable. A moderately relevant object, an object with mixed qualities, or an irrelevant but harmless background element. A score of 0.0 is perfectly neutral.
    * **-0.9 to -0.5:** Bad. The object is irrelevant and distracting, or it is a relevant object with severe visual artifacts/flaws.
    * **-1.0:** Very Bad. The object actively undermines the image and directly contradicts the prompt's intent.

* **Output Lines:**
    `Score: [A single number between -1.0 and 1.0]`
---
## EXAMPLE OUTPUT STRUCTURE
**Region ID: 1**
Identified Object: A running golden retriever.
Score: 0.95
---
**Region ID: 2**
Identified Object: A tall green tree in the background.
Score: 0.2
---
**Region ID: 3**
Identified Object: A distorted red shape on the far left.
Score: -0.8
---
Begin your analysis now.
"""

def parse_vlm_evaluation_output(vlm_text_output: str) -> dict:
    """
    Parses the VLM's structured output for multiple regions.
    """
    evaluations = {}
    content = vlm_text_output.split('</think>')[-1].strip()
    blocks = content.split('\n\n')

    for block in blocks:
        block = block.strip()
        if not block: continue
        try:
            id_match = re.search(r"Region ID:\s*(\d+)", block, re.IGNORECASE)
            if not id_match: continue
            region_id = int(id_match.group(1))

            evaluations[region_id] = {}
            id_obj_match = re.search(r"Identified Object:\s*(.*)", block, re.IGNORECASE)
            score_match = re.search(r"Score:\s*(-?\d+\.?\d*)", block, re.IGNORECASE)
            reason_match = re.search(r"Reason:\s*(.*)", block, re.IGNORECASE | re.DOTALL)
            
            if id_obj_match:
                evaluations[region_id]['identified_object'] = id_obj_match.group(1).strip()
            if score_match:
                evaluations[region_id]['score'] = float(score_match.group(1))
            if reason_match:
                evaluations[region_id]['reason'] = reason_match.group(1).strip()

        except (ValueError, IndexError, AttributeError) as e:
            logging.warning(f"Failed to parse block for Region ID {id_match.group(1) if id_match else 'Unknown'}. Error: {e}\nBlock:\n{block}")
            continue
            
    return evaluations


# --- 2. Helper Function for SAM ---

def get_all_bboxes_with_sam(sam_mask_generator, image: Image.Image) -> list[dict]:
    """
    Uses Segment Anything Model (SAM) to automatically detect all objects in the image.
    """
    logging.info("Running SAM for unsupervised object detection...")
    image_np_rgb = np.array(image.convert("RGB"))
    masks = sam_mask_generator.generate(image_np_rgb)

    detected_objects = []
    if not masks:
        logging.warning("SAM did not find any objects.")
        return []

    for mask_info in masks:
        box_xywh = mask_info['bbox']
        x1, y1, w, h = box_xywh[0], box_xywh[1], box_xywh[2], box_xywh[3]
        detected_objects.append({
            "bbox": [int(x1), int(y1), int(x1 + w), int(y1 + h)],
            "area": int(mask_info['area']),
        })
        
    detected_objects.sort(key=lambda x: x['area'], reverse=True)
    logging.info(f"SAM found {len(detected_objects)} objects.")
    return detected_objects


# --- 3. Main Program ---

def main(args):
    # --- VLM Model Loading ---
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', 0)
    logging.info(f"[GPU {gpu_id}] Loading BAGEL VLM from {args.model_path}...")
    model_path = args.model_path
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    config = BagelConfig(visual_gen=True, visual_und=True, llm_config=llm_config, vit_config=vit_config, vae_config=vae_config, vit_max_num_patch_per_side=70, connector_act='gelu_pytorch_tanh', latent_patch_size=2, max_latent_size=64)
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)
    device_map = infer_auto_device_map(model, max_memory={i: args.max_mem_per_gpu for i in range(torch.cuda.device_count())}, no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"])
    same_device_modules = ['language_model.model.embed_tokens', 'time_embedder', 'latent_pos_embed', 'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed']
    if torch.cuda.device_count() > 0:
        first_device = device_map.get(same_device_modules[0], f"cuda:{torch.cuda.current_device()}")
        for k in same_device_modules: device_map[k] = device_map.get(k, first_device)
    filename = "ema.safetensors" if "BAGEL-7B-MoT" in model_path else "model.safetensors"
    checkpoint_path = os.path.join(model_path, filename)
    logging.info(f"Using device map: {device_map}")
    model = load_checkpoint_and_dispatch(model, checkpoint=checkpoint_path, device_map=device_map, offload_buffers=True, dtype=torch.bfloat16, force_hooks=True, offload_folder="/tmp/offload")
    model = model.eval()
    inferencer = InterleaveInferencer(model=model, vae_model=vae_model, tokenizer=tokenizer, vae_transform=vae_transform, vit_transform=vit_transform, new_token_ids=new_token_ids)
    logging.info(f"[GPU {gpu_id}] BAGEL VLM loaded successfully.")

    # --- Conditionally load SAM model ---
    sam_mask_generator = None
    # SAM is needed for region-based analysis. If --sample_level_only is NOT set, and we're not just doing global reward, load SAM.
    if not args.sample_level_only:
        if not args.sam_model_path or not os.path.exists(args.sam_model_path):
            raise FileNotFoundError(f"SAM model needed for region-based analysis. Not found at: {args.sam_model_path}.")
        logging.info("Loading SAM model for region-based detection...")
        device = next(inferencer.model.parameters()).device
        sam_model = sam_model_registry[args.sam_model_type](checkpoint=args.sam_model_path).to(device)
        sam_mask_generator = SamAutomaticMaskGenerator(sam_model)
        logging.info("SAM model loaded successfully.")

    # --- Data Processing and Inference Loop ---
    if not os.path.exists(args.input_jsonl):
        raise FileNotFoundError(f"Input file not found: {args.input_jsonl}")

    records = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning(f"Skipping malformed JSON line.")

    write_mode = "w" if args.overwrite else "a"
    with open(args.output_jsonl, write_mode, encoding="utf-8") as out_f:
        for record in tqdm(records, desc=f"Analyzing Images on GPU {gpu_id}"):
            image_file = record.get('image_file')
            prompt = record.get('original_prompt')

            if not image_file or not prompt:
                logging.warning(f"Skipping record due to missing 'image_file' or 'prompt': {record}")
                continue

            image_path = os.path.join(args.image_dir, image_file)
            if not os.path.exists(image_path):
                logging.warning(f"Image file not found: {image_path}")
                continue
            
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logging.warning(f"Failed to open image {image_path}: {e}")
                continue
            
            final_rewards = []
            vlm_full_output = ""
            width, height = image.size
            
            # --- Main workflow branching ---
            if args.sample_level_only:
                # --- SAMPLE-LEVEL WORKFLOW (FULL IMAGE) ---
                instruction_prompt = create_sample_level_evaluation_prompt(prompt)
                parsed_evaluation = {}
                
                for attempt in range(args.max_retries):
                    output_dict = inferencer(image=image, text=instruction_prompt, think=True, understanding_output=True, max_think_token_n=1024, text_temperature=0.0)
                    vlm_full_output = output_dict.get('text', '')
                    parsed_evaluation = parse_sample_level_output(vlm_full_output)
                    if 'score' in parsed_evaluation:
                        logging.info(f"Successfully parsed sample-level VLM output for {image_file} on attempt {attempt + 1}.")
                        break
                    else:
                        logging.warning(f"Attempt {attempt+1}/{args.max_retries} for {image_file} (sample-level) did not yield a parsable score. Retrying...")

                if 'score' in parsed_evaluation:
                    reward_info = parsed_evaluation
                    reward_info['bbox'] = [0, 0, width, height]
                    reward_info['area'] = width * height
                    if args.binarize_score:
                        reward_info['binarized_score'] = 1 if reward_info['score'] > 0 else -1
                    final_rewards.append(reward_info)
                else:
                    logging.warning(f"All VLM attempts failed for {image_file}. Assigning a default reward of 0.")
                    default_reward = {'bbox': [0, 0, width, height], 'area': width * height, 'score': 0.0, 'reason': f"VLM failed after {args.max_retries} attempts."}
                    if args.binarize_score:
                        default_reward['binarized_score'] = -1
                    final_rewards.append(default_reward)
            
            else:
                # --- REGION-BASED WORKFLOW (PER-OBJECT) ---
                if not sam_mask_generator:
                    logging.error("SAM model was not loaded. Cannot perform region-based analysis. Please run without --sample_level_only.")
                    continue

                all_detected_objects = get_all_bboxes_with_sam(sam_mask_generator, image)
                objects_to_evaluate = all_detected_objects[:args.max_regions]

                if not objects_to_evaluate:
                    logging.warning(f"SAM found no objects for {image_file}. Skipping region-based analysis.")
                else:
                    instruction_prompt = create_hybrid_evaluation_prompt(prompt, objects_to_evaluate)
                    parsed_evaluations = {}
                    for attempt in range(args.max_retries):
                        output_dict = inferencer(image=image, text=instruction_prompt, think=True, understanding_output=True, max_think_token_n=4096, text_temperature=0.0)
                        vlm_full_output = output_dict.get('text', '')
                        parsed_evaluations = parse_vlm_evaluation_output(vlm_full_output)
                        if parsed_evaluations:
                            logging.info(f"Successfully parsed VLM output for {image_file} on attempt {attempt + 1}.")
                            break
                        else:
                            logging.warning(f"Attempt {attempt+1}/{args.max_retries} for {image_file} did not yield any parsable regions. Retrying...")
                    
                    if not parsed_evaluations:
                        logging.warning(f"All {args.max_retries} VLM attempts failed for {image_file}. Skipping region-based rewards.")
                    else:
                        for obj in objects_to_evaluate:
                            region_id = obj.get('region_id')
                            if region_id in parsed_evaluations and 'score' in parsed_evaluations[region_id]:
                                reward_info = parsed_evaluations[region_id]
                                if args.binarize_score:
                                    original_score = reward_info.get('score', 0.0)
                                    reward_info['binarized_score'] = 1 if original_score > 0 else -1
                                reward_info['bbox'] = obj['bbox']
                                reward_info['area'] = obj['area']
                                final_rewards.append(reward_info)

            # --- GLOBAL LAYOUT REWARD WORKFLOW (NEW) ---
            global_layout_reward = None
            if args.global_layout_reward:
                logging.info(f"Performing global layout analysis for {image_file}...")
                instruction_prompt_global = create_global_layout_reward_prompt(prompt)
                parsed_global_evaluation = {}
                
                for attempt in range(args.max_retries):
                    output_dict_global = inferencer(image=image, text=instruction_prompt_global, think=True, understanding_output=True, max_think_token_n=1024, text_temperature=0.0)
                    vlm_full_output_global = output_dict_global.get('text', '')
                    parsed_global_evaluation = parse_global_layout_output(vlm_full_output_global)
                    if 'global_layout_score' in parsed_global_evaluation:
                        logging.info(f"Successfully parsed global layout VLM output for {image_file} on attempt {attempt + 1}.")
                        break
                    else:
                        logging.warning(f"Attempt {attempt+1}/{args.max_retries} for {image_file} (global layout) did not yield a parsable score. Retrying...")

                if 'global_layout_score' in parsed_global_evaluation:
                    global_layout_reward = parsed_global_evaluation
                    global_layout_reward['bbox'] = [0, 0, width, height]
                    global_layout_reward['area'] = width * height
                else:
                    logging.warning(f"All VLM attempts failed for global layout on {image_file}. Assigning a default score of 0.")
                    global_layout_reward = {
                        'bbox': [0, 0, width, height],
                        'area': width * height,
                        'global_layout_score': 0.0,
                        'reason': f"VLM failed to produce a parsable output after {args.max_retries} attempts."
                    }

            # --- Final Record Assembly ---
            new_record = {
                "image_file": image_file,
                "prompt": prompt,
                "vlm_rewards": final_rewards,
                "global_layout_reward": global_layout_reward,
                "vlm_raw_output": vlm_full_output
            }
            out_f.write(json.dumps(new_record, ensure_ascii=False) + "\n")

    logging.info(f"Analysis complete. Results saved to {args.output_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze images with a VLM using region-based or sample-level evaluation.")
    
    # --- Workflow arguments ---
    parser.add_argument("--sample_level_only", action="store_true",
                        help="If set, performs a single evaluation on the entire image, ignoring regions. The default is region-based analysis.")
    parser.add_argument("--global_layout_reward", action="store_true",
                        help="If set, adds a separate global reward that evaluates only the layout and positional relationships of objects, in addition to the main evaluation.")
    parser.add_argument("--sam_model_path", type=str, default="sam_vit_h_4b8939.pth",
                        help="Path to the SAM model checkpoint file (required for region-based analysis).")
    parser.add_argument("--sam_model_type", type=str, default="vit_h", help="The type of SAM model (e.g., 'vit_h', 'vit_l', 'vit_b').")
    parser.add_argument("--max_regions", type=int, default=5, help="Max number of regions to evaluate (region-based mode only).")

    # --- Core model and data arguments ---
    parser.add_argument("--model_path", type=str, required=True, help="Path to the BAGEL model checkpoint directory.")
    parser.add_argument("--max_mem_per_gpu", type=str, default="80GiB", help="Maximum memory per GPU (e.g., '40GiB').")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Input JSONL file with image metadata.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory where the images are stored.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to write the final JSONL output.")
    
    # --- Control arguments ---
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries if VLM output parsing fails.")
    parser.add_argument("--binarize_score", action="store_true", 
                        help="If set, adds a 'binarized_score' field (1 for score > 0, -1 otherwise).")
    
    args = parser.parse_args()
    
    # Ensure SAM dependencies are available if needed.
    if (not args.sample_level_only and not args.global_layout_reward) and not UNSUPERVISED_DEPS_AVAILABLE:
        raise ImportError("Dependencies for region-based mode (segment-anything-py, opencv-python) are not installed. Please run 'pip install segment-anything-py opencv-python' or use the --sample_level_only flag.")

    main(args)