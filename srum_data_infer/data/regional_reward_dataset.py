# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import numpy as np
from PIL import Image
import torch
import random
from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
import os

Image.MAX_IMAGE_PIXELS = None # Adjust as needed

class RegionalRewardDataset(DistributedIterableDataset):
    """
    An iterable dataset for handling images with regional reward annotations,
    using a "good" image as the ground truth and a "bad" image as the input.
    It reads from a JSONL file where each line contains paths to both images,
    a prompt, and VLM-generated regional rewards for the bad image.
    """
    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        jsonl_path,
        image_dirs,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        **kwargs,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.image_dirs = image_dirs
        self.data_status = data_status
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
        
        num_lines_per_rank = len(self.lines) // self.world_size
        local_start = self.local_rank * num_lines_per_rank
        local_end = (self.local_rank + 1) * num_lines_per_rank
        self.lines_per_rank = self.lines[local_start:local_end]

    def _parse_vlm_output_to_reward_map(self, vlm_rewards, image_size, global_reward_factor=0.0):
        """
        Parses the VLM's structured output to generate a 2D reward map.
        The reward map is then scaled by a global reward factor.
        
        Args:
            vlm_rewards (list): A list of dictionaries containing regional reward information.
            image_size (tuple): The (width, height) of the image.
            global_reward_factor (float): A scaling factor from the global layout reward.
                                          This value should be between -1.0 and 1.0.
        """
        width, height = image_size
        reward_map = np.full((height, width), 0.5, dtype=np.float32)

        if not vlm_rewards:
            # If no regional rewards, scale the neutral map by the global factor
            factor = (global_reward_factor + 1.0) / 2.0
            reward_map = reward_map * factor
            return torch.from_numpy(reward_map)
        
        # Apply regional rewards
        for item in vlm_rewards:
            # The global reward is handled separately, so we explicitly skip it here.
            if item.get('object') == 'global_layout_reward':
                continue

            score = item.get('score', 0.0)
            bbox = item.get('bbox')
            
            # ROBUSTNESS: Validate bbox structure and values
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue # Skip if bbox is missing, not a list/tuple, or not of length 4

            try:
                x_min, y_min, x_max, y_max = map(int, bbox)
            except (ValueError, TypeError):
                print(f"Warning: Skipping region with non-numeric coordinates in bbox {bbox}.")
                continue # Skip if coordinates are not valid numbers
            
            if x_min >= x_max or y_min >= y_max:
                print(f"Warning: Skipping region with invalid coordinates (min >= max) in bbox {bbox}.")
                continue # Skip if coordinates are illogical

            # Scale regional score to a weight between 0 and 1
            weight = (score + 1.0) / 2.0
            
            # Clamp coordinates to be within image bounds
            x_min_clamped, x_max_clamped = max(0, x_min), min(width, x_max)
            y_min_clamped, y_max_clamped = max(0, y_min), min(height, y_max)
            
            # Apply weight to the valid region slice
            if x_min_clamped < x_max_clamped and y_min_clamped < y_max_clamped:
                region_slice = reward_map[y_min_clamped:y_max_clamped, x_min_clamped:x_max_clamped]
                np.maximum(region_slice, weight, out=region_slice)
            
        # Apply the global reward factor to the entire reward map
        # Shift score from [-1, 1] to [0, 2] and then divide by 2 to get a factor [0, 1].
        factor = (global_reward_factor + 1.0) / 2.0
        # print(factor,"factor")
        reward_map = reward_map * factor

        return torch.from_numpy(reward_map)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1
        
        lines_per_worker = self.lines_per_rank[worker_id::num_workers]

        start_line_idx = 0
        if self.data_status is not None and worker_id in self.data_status:
            start_line_idx = self.data_status[worker_id][0] + 1
        
        if self.local_rank == 0 and worker_id == 0:
            print(
                f"Rank-{self.local_rank} Worker-{worker_id} Dataset-{self.dataset_name}: "
                f"Total lines for this rank: {len(self.lines_per_rank)}. "
                f"Total lines for this worker: {len(lines_per_worker)}. "
                f"Resuming from line index: {start_line_idx}"
            )

        while True:
            random.shuffle(lines_per_worker)
            current_iter_lines = lines_per_worker[start_line_idx:]

            for line in current_iter_lines:
                try:
                    global_line_idx = self.lines.index(line)
                except ValueError:
                    continue

                try:
                    data = json.loads(line)
                    bad_image_path = os.path.join(self.image_dirs['bad'], data['bad_image_file'])
                    good_image_path = os.path.join(self.image_dirs['good'], data['good_image_file'])
                    
                    original_prompt = data.get('prompt', "") # Use .get for safety
                    all_rewards = data.get('vlm_rewards', [])
                    
                    # ROBUSTNESS: Explicitly find global reward and handle its absence
                    global_reward_score = None
                    for reward in all_rewards:
                        if reward.get('object') == 'global_layout_reward':
                            global_reward_score = reward.get('score')
                            break # Found it, no need to look further
                    
                    if global_reward_score is None:
                        # If not found or score key is missing, use a neutral default
                        print(f"Warning: 'global_layout_reward' score not found for line {global_line_idx}. Using neutral score 0.0.")
                        global_reward_score = 0.0

                    # Filter out the global reward to get only regional rewards
                    vlm_regional_rewards = [r for r in all_rewards if r.get('object') != 'global_layout_reward']

                    bad_image = pil_img2rgb(Image.open(bad_image_path))
                    good_image = pil_img2rgb(Image.open(good_image_path))

                    bad_image_tensor = self.transform(bad_image)
                    good_image_tensor = self.transform(good_image)
                    
                    height, width = bad_image_tensor.shape[1:]
                    
                    # Pass global_reward_score to the reward map function
                    reward_map_tensor = self._parse_vlm_output_to_reward_map(
                        vlm_regional_rewards, (width, height), global_reward_factor=global_reward_score
                    )

                    caption_token = self.tokenizer.encode(original_prompt)
                    num_tokens = len(caption_token) + (width * height // (self.transform.stride if hasattr(self.transform, 'stride') else 16**2))
                    
                    sequence_plan = [
                        {'type': 'text', 'enable_cfg': 1, 'loss': 0},
                        {'type': 'vae_image', 'enable_cfg': 0, 'loss': 1}
                    ]
                    
                    sample = dict(
                        image_tensor_list=[bad_image_tensor], # Input to the model
                        gt_image_tensor=good_image_tensor,    # Ground truth for loss
                        text_ids_list=[caption_token],
                        num_tokens=num_tokens,
                        sequence_plan=sequence_plan,
                        regional_reward_map=reward_map_tensor,
                        data_indexes={
                            "dataset_name": self.dataset_name,
                            "worker_id": worker_id,
                            "data_indexes": [global_line_idx], 
                        }
                    )
                    yield sample

                except Exception as e:
                    print(f"Error processing line_idx {global_line_idx} in {self.dataset_name}: {e}")
                    continue
            
            start_line_idx = 0