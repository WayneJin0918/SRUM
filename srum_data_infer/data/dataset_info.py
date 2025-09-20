# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset
from .regional_reward_dataset import RegionalRewardDataset
DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
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
            'num_total_samples': 292, # total number of samples in dataset
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
            'num_total_samples': 296, # total number of samples in dataset
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
            'num_total_samples': 395, # total number of samples in dataset
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
            'num_total_samples': 5911, # total number of samples in dataset
        },
    }
}