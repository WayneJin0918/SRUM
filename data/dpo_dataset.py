# dpo_dataset.py (完全替换)

import json
import os
from PIL import Image
import torch

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset

class DPODataset(DistributedIterableDataset):
    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        jsonl_path,
        chosen_images_dir,
        rejected_images_dir,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        shuffle_lines=False,
        vae_image_downsample=16, # ✅ 新增：从配置中接收VAE下采样率
        **kwargs
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.jsonl_path = jsonl_path
        self.chosen_images_dir = chosen_images_dir
        self.rejected_images_dir = rejected_images_dir
        self.shuffle_lines = shuffle_lines
        self.vae_image_downsample = vae_image_downsample # ✅ 新增：保存下采样率

        # 建议：对于超大数据集，不要在此处一次性加载所有jsonl，应在 __iter__ 中流式读取
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            self._all_jsonl_data = [json.loads(line) for line in f]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers_per_rank = worker_info.num_workers if worker_info else 1

        # 数据分片逻辑 (简化版)
        indices = list(range(len(self._all_jsonl_data)))
        if self.shuffle_lines:
            import random
            random.Random(42).shuffle(indices) # 注意：固定种子会导致每个epoch顺序相同

        worker_indices = indices[self.local_rank * num_workers_per_rank + worker_id :: self.world_size * num_workers_per_rank]

        for idx in worker_indices:
            data_item = self._all_jsonl_data[idx]

            prompt_text = data_item['original_prompt']
            image_file = data_item['image_file']

            chosen_image_path = os.path.join(self.chosen_images_dir, image_file)
            rejected_image_path = os.path.join(self.rejected_images_dir, image_file)

            try:
                text_ids = self.tokenizer.encode(prompt_text)
                chosen_image = self.transform(pil_img2rgb(Image.open(chosen_image_path)))
                rejected_image = self.transform(pil_img2rgb(Image.open(rejected_image_path)))
                
                # ✅ 关键修改：计算 num_tokens
                num_text_tokens = len(text_ids)
                # 使用 chosen_image 的 shape 来计算图像 token 数 (chosen 和 rejected 的 shape 应该一样)
                H, W = chosen_image.shape[1:]
                h = H // self.vae_image_downsample
                w = W // self.vae_image_downsample
                num_image_tokens = h * w
                num_tokens = num_text_tokens + num_image_tokens

            except Exception as e:
                print(f"Worker {worker_id} skipping DPO sample {idx} due to loading error: {e}")
                continue
            
            sequence_plan = [
                {'type': 'text', 'enable_cfg': 1, 'loss': 0, 'special_token_loss': 0},
                {'type': 'vae_image', 'enable_cfg': 0, 'loss': 1, 'special_token_loss': 0},
            ]

            # 产出 "chosen" 样本
            yield {
                'num_tokens': num_tokens, # ✅ 新增
                'image_tensor_list': [chosen_image],
                'text_ids_list': [text_ids],
                'sequence_plan': sequence_plan,
                'dpo_info': {'prompt_id': idx, 'type': 'chosen'},
                'data_indexes': {"dataset_name": self.dataset_name}
            }

            # 产出 "rejected" 样本
            yield {
                'num_tokens': num_tokens, # ✅ 新增
                'image_tensor_list': [rejected_image],
                'text_ids_list': [text_ids],
                'sequence_plan': sequence_plan,
                'dpo_info': {'prompt_id': idx, 'type': 'rejected'},
                'data_indexes': {"dataset_name": self.dataset_name}
            }