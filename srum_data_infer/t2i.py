import os
import re
import json
import argparse
from PIL import Image
import torch
import numpy as np
import random

# Assuming all other necessary imports from the original script are here
from copy import deepcopy
from typing import (Any, AsyncIterable, Callable, Dict, Generator, List, NamedTuple, Optional, Tuple, Union)
import requests
from io import BytesIO
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from safetensors.torch import load_file
from inferencer import InterleaveInferencer

# The parser is now defined globally so the launcher can import it
parser = argparse.ArgumentParser(description="Generate images using BAGEL model.")
parser.add_argument("--model_path", type=str, default="/mnt/data/checkpoints/BAGEL-7B-MoT", help="Path to the BAGEL model checkpoint directory.")
parser.add_argument("--max_mem_per_gpu", type=str, default="80GiB", help="Maximum memory per GPU (e.g., '40GiB').")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
parser.add_argument("--metadata_file", type=str, required=True, help="JSON file with prompts.")
parser.add_argument("--think", action="store_true", help="Enable 'think' mode for detailed prompt generation.")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing image files.")
parser.add_argument("--num_timesteps", type=int, default=100)
parser.add_argument("--jsonl_output_path", type=str, default=None, help="Path to write the structured JSONL output.")

def extract_detailed_prompt(think_output):
    """From think_output, extract content after 'Here’s the finished detailed prompt:'"""
    pattern = r"detailed prompt:(.*?)(?=(\n\n|\Z))"
    match = re.search(pattern, think_output, re.DOTALL)
    return match.group(1).strip() if match else "Failed to extract detailed prompt."

def main(args):
    # --- Model Loading and Setup (remains mostly unchanged) ---
    model_path = args.model_path
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
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
    max_mem_per_gpu = args.max_mem_per_gpu
    device_map = infer_auto_device_map(model, max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())}, no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"])
    print(device_map)
    same_device_modules = ['language_model.model.embed_tokens', 'time_embedder', 'latent_pos_embed', 'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed']
    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules: device_map[k] = device_map.get(k, first_device)
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules: device_map[k] = device_map.get(k, first_device)
    # print(model_path)
    # assert 0
    filename = "ema.safetensors" if "BAGEL-7B-MoT" in model_path else "model.safetensors"

    # Construct the full path to the checkpoint file
    checkpoint_path = os.path.join(model_path, filename)

    # Call the function once with the correct checkpoint path
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_path,
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload"
    )
    model = model.eval()
    print('Model loaded')
    inferencer = InterleaveInferencer(model=model, vae_model=vae_model, tokenizer=tokenizer, vae_transform=vae_transform, vit_transform=vit_transform, new_token_ids=new_token_ids)
    # --- End of Model Loading ---

    # Set random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output images will be saved in {args.output_dir}")

    with open(args.metadata_file, "r") as f:
        metadatas = json.load(f)
    
    for idx, metadata in enumerate(metadatas):
        prompt = metadata['Question']
        # Use the pre-assigned prompt_id from the launcher
        prompt_id = metadata['Question_id']
        # The original_index is needed for sorting later by the launcher
        original_index = metadata['original_index']
        
        outpath = os.path.join(args.output_dir, f"{prompt_id}.png")

        print(f"Processing prompt {idx + 1}/{len(metadatas)} (ID: {prompt_id}): '{prompt}'")

        if os.path.exists(outpath) and not args.overwrite:
            print(f"Skipping generation for ID: {prompt_id} (file already exists).")
            continue

        if args.think:
            inference_hyper = dict(max_think_token_n=1000, do_sample=False, cfg_text_scale=4.0, cfg_img_scale=1.0, cfg_interval=[0.4, 1.0], timestep_shift=3.0, num_timesteps=args.num_timesteps, cfg_renorm_min=0.0, cfg_renorm_type="global")
            
            output_dict = inferencer(text=prompt, think=True, **inference_hyper)
            think_output = output_dict.get('text', '')
            detailed_prompt = extract_detailed_prompt(think_output)
            
            # Create a record including the original_index for sorting
            record = {
                "original_index": original_index,
                "prompt_id": prompt_id,
                "image_file": os.path.basename(outpath),
                "original_prompt": prompt,
                "detailed_prompt": detailed_prompt,
                "think_output": think_output
            }
            
            # Write to the specific temporary file provided by the launcher
            if args.jsonl_output_path:
                with open(args.jsonl_output_path, "a", encoding="utf-8") as jsonl_file:
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        else: # Normal inference mode
            inference_hyper = dict(cfg_text_scale=4.0, cfg_img_scale=1.0, cfg_interval=[0.4, 1.0], timestep_shift=3.0, num_timesteps=args.num_timesteps, cfg_renorm_min=0.0, cfg_renorm_type="global")
            output_dict = inferencer(text=prompt, **inference_hyper)

        # Save image (this part is common to both modes)
        # Check if an image was actually generated
        if 'image' in output_dict and isinstance(output_dict['image'], Image.Image):
            tmpimage = output_dict['image']
            if tmpimage.getbbox(): # Crop blank border if image is not empty
                tmpimage = tmpimage.crop(tmpimage.getbbox())
            tmpimage.save(outpath)
            print(f"Image saved to {outpath}")
        else:
            print(f"Warning: No image was generated for prompt ID {prompt_id}.")
        
        print('-' * 20)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)