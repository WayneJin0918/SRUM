# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset
from .dpo_dataset import DPODataset # Import your new DPO dataset class
from .regional_reward_dataset import RegionalRewardDataset
DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'dpo_image_gen': DPODataset,
    # === 新增 ===
    'regional_reward': RegionalRewardDataset,
    # ============
}



DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': 'images_wise/bagel_base_7b_300_step_think_parquet', # path of the parquet files
            'num_files': 1, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
        'comp_data_0p1': {
            'data_dir': 'images_comp/bagel_base_7b_300_step_think_parquet', # path of the parquet files
            'num_files': 1, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 625, # number of total samples in the dataset
        },
        'comp_data_0p5': {
            'data_dir': 'images_comp/bagel_base_7b_300_step_think_parquet', # path of the parquet files
            'num_files': 1, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 3117, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'your_data_path/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'your_data_path/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', 
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': 'your_data_path/bagel_example/vlm/images',
			'jsonl_path': 'your_data_path/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
        'lb_txt': {
            'data_dir': '/mnt/data/nyw/Bagel/bagel_example/vlm/images',
			'jsonl_path': '/mnt/data/nyw/Bagel/bagel_example/vlm/linbin_txt.jsonl',
			'num_total_samples': 64
        }
    },
    'dpo_image_gen': { # New entry for DPO dataset
        'my_dpo_data': { # You can choose a descriptive name for this specific DPO dataset instance
            'jsonl_path': 'images_wise/bagel_base_7b_300_step_think/output.jsonl',
            'image_dirs': {
                'positive': 'images_wise/bagel_base_7b_300_step_think', # 替换为您的实际路径
                'negative': 'images_wise/bagel_base_7b_50_step_no_think', # 替换为您的实际路径
            },
            'chosen_images_dir': 'images_wise/bagel_base_7b_300_step_think',
            'rejected_images_dir': 'images_wise/bagel_base_7b_50_step_no_think',
            'num_files': 1, # Since it's one JSONL file, conceptually '1'. Adjust if your sharding needs differ.
            'num_total_samples': 1000, # Approximate total number of samples in your JSONL, crucial for distributed training,
            'data_dir': 'images_base_dir', # **新增或修改此行**,
        },
    },
    # ======================= MODIFICATION START =======================
    'regional_reward': {
        'st_wise': {
            'jsonl_path': 'wise_sub/wise_sub_image/spatio-temporal_reasoning_base_7b_300_step_think_regional_rewards.jsonl',
            # Replace 'image_base_dir' with the 'image_dirs' dictionary
            'image_dirs': {
                # Key 'good' for the ground-truth images
                'good': 'wise_sub/wise_sub_image/spatio-temporal_reasoning_base_7b_300_step_think',
                # Key 'bad' for the input images that have rewards
                'bad': 'wise_sub/wise_sub_image/spatio-temporal_reasoning_base_7b_300_step_think' 
            },
            'num_total_samples': 292, # 你的数据集总样本数
        },
        'ns_wise': {
            'jsonl_path': 'wise_sub/wise_sub_image/natural_science_base_7b_300_step_think_regional_rewards.jsonl',
            # Replace 'image_base_dir' with the 'image_dirs' dictionary
            'image_dirs': {
                # Key 'good' for the ground-truth images
                'good': 'wise_sub/wise_sub_image/natural_science_base_7b_300_step_think',
                # Key 'bad' for the input images that have rewards
                'bad': 'wise_sub/wise_sub_image/natural_science_base_7b_300_step_think' 
            },
            'num_total_samples': 296, # 你的数据集总样本数
        },
        'cul_wise': {
            'jsonl_path': 'wise_sub/wise_sub_image/cultural_common_sense_base_7b_300_step_think_regional_rewards.jsonl',
            # Replace 'image_base_dir' with the 'image_dirs' dictionary
            'image_dirs': {
                # Key 'good' for the ground-truth images
                'good': 'wise_sub/wise_sub_image/cultural_common_sense_base_7b_300_step_think',
                # Key 'bad' for the input images that have rewards
                'bad': 'wise_sub/wise_sub_image/cultural_common_sense_base_7b_300_step_think' 
            },
            'num_total_samples': 395, # 你的数据集总样本数
        },
        'comp_data': {
            'jsonl_path': 'images_comp/images_comp_base_7b_300_step_think_regional_rewards.jsonl',
            # Replace 'image_base_dir' with the 'image_dirs' dictionary
            'image_dirs': {
                # Key 'good' for the ground-truth images
                'good': 'images_comp/bagel_base_7b_300_step_think',
                # Key 'bad' for the input images that have rewards
                'bad': 'images_comp/bagel_base_7b_300_step_think' 
            },
            'num_total_samples': 5911, # 你的数据集总样本数
        },
        'comp_data_0p1_glo_and_regional': {
            'jsonl_path': 'images_comp/images_comp_base_sub_0p1_7b_300_step_think_glo_and_regional_rewards.jsonl',
            # Replace 'image_base_dir' with the 'image_dirs' dictionary
            'image_dirs': {
                # Key 'good' for the ground-truth images
                'good': 'images_comp/bagel_base_7b_300_step_think',
                # Key 'bad' for the input images that have rewards
                'bad': 'images_comp/bagel_base_7b_300_step_think' 
            },
            'num_total_samples': 690, # 你的数据集总样本数
        },
        'comp_data_0p5': {
            'jsonl_path': 'images_comp/images_comp_base_sub_0p5_7b_300_step_think_regional_rewards.jsonl',
            # Replace 'image_base_dir' with the 'image_dirs' dictionary
            'image_dirs': {
                # Key 'good' for the ground-truth images
                'good': 'images_comp/bagel_base_7b_300_step_think',
                # Key 'bad' for the input images that have rewards
                'bad': 'images_comp/bagel_base_7b_300_step_think' 
            },
            'num_total_samples': 3006, # 你的数据集总样本数
        },
        'comp_data_0p1_sample': {
            'jsonl_path': 'images_comp/images_comp_base_sub_0p1_300_step_think_sample_level_rewards.jsonl',
            # Replace 'image_base_dir' with the 'image_dirs' dictionary
            'image_dirs': {
                # Key 'good' for the ground-truth images
                'good': 'images_comp/bagel_base_7b_300_step_think',
                # Key 'bad' for the input images that have rewards
                'bad': 'images_comp/bagel_base_7b_300_step_think' 
            },
            'num_total_samples': 480, # 你的数据集总样本数
        },
        'comp_data_0p1_sample_bi': {
            'jsonl_path': 'images_comp/images_comp_base_sub_0p1_300_step_think_sample_level_binarized_rewards.jsonl',
            # Replace 'image_base_dir' with the 'image_dirs' dictionary
            'image_dirs': {
                # Key 'good' for the ground-truth images
                'good': 'images_comp/bagel_base_7b_300_step_think',
                # Key 'bad' for the input images that have rewards
                'bad': 'images_comp/bagel_base_7b_300_step_think' 
            },
            'num_total_samples': 480, # 你的数据集总样本数
        },
        'comp_data_0p1_reg_bi': {
            'jsonl_path': 'images_comp/images_comp_base_sub_0p1_300_step_think_regional_binarized_rewards.jsonl',
            # Replace 'image_base_dir' with the 'image_dirs' dictionary
            'image_dirs': {
                # Key 'good' for the ground-truth images
                'good': 'images_comp/bagel_base_7b_300_step_think',
                # Key 'bad' for the input images that have rewards
                'bad': 'images_comp/bagel_base_7b_300_step_think' 
            },
            'num_total_samples': 1216, # 你的数据集总样本数
        },
    }
    # ======================= MODIFICATION END =======================
}