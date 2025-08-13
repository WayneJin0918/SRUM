python vlm_multi_gpu.py \
    --model_path BAGEL-7B-MoT \
    --image_dir images_comp/bagel_base_7b_300_step_think \
    --input_jsonl images_comp/bagel_base_7b_300_step_think/output.jsonl \
    --output_jsonl regional_rewards/images_comp_base_7b_300_step_think_regional_rewards.jsonl \
    --num_gpus 8 \
    --processes_per_gpu 2 \

python vlm_format_output.py \
    --input_jsonl regional_rewards/images_comp_base_7b_300_step_think_regional_rewards.jsonl \
    --output_jsonl images_comp/images_comp_base_7b_300_step_think_regional_rewards.jsonl
    
    
# python vlm_multi_gpu.py \
#     --model_path BAGEL-7B-MoT \
#     --image_dir images_comp/bagel_base_7b_300_step_think \
#     --input_jsonl images_comp/bagel_base_7b_300_step_think/output_sub_0p1.jsonl \
#     --output_jsonl sample_level_rewards/images_comp_base_sub_0p1_300_step_think_sample_level_rewards.jsonl \
#     --num_gpus 8 \
#     --processes_per_gpu 2 \
#     --sample_level_only \

# # 步骤 2: (可选) 格式化 sample-level 的输出结果
# python vlm_format_output.py \
#     --input_jsonl sample_level_rewards/images_comp_base_sub_0p1_300_step_no_think_sample_level_rewards.jsonl \
#     --output_jsonl images_comp/images_comp_base_sub_0p1_300_step_think_sample_level_rewards.jsonl

# python vlm_multi_gpu.py \
#     --model_path BAGEL-7B-MoT \
#     --image_dir images_comp/bagel_base_7b_300_step_think \
#     --input_jsonl images_comp/bagel_base_7b_300_step_think/output_sub_0p1.jsonl \
#     --output_jsonl sample_level_rewards/images_comp_base_sub_0p1_300_step_think_sample_level_binarized_rewards.jsonl \
#     --num_gpus 8 \
#     --processes_per_gpu 2 \
#     --sample_level_only \
#     --binarize_score

# # 步骤 2: (可选) 格式化 sample-level 的输出结果
# python vlm_format_output.py \
#     --input_jsonl sample_level_rewards/images_comp_base_sub_0p1_300_step_no_think_sample_level_binarized_rewards.jsonl \
#     --output_jsonl images_comp/images_comp_base_sub_0p1_300_step_think_sample_level_binarized_rewards.jsonl

# # 步骤 1: 使用 vlm_multi_gpu.py 进行 Region-Based 分析并增加二值化分数
# python vlm_multi_gpu.py \
#     --model_path BAGEL-7B-MoT \
#     --image_dir images_comp/bagel_base_7b_300_step_think \
#     --input_jsonl images_comp/bagel_base_7b_300_step_think/output_sub_0p1.jsonl \
#     --output_jsonl regional_rewards/images_comp_base_sub_0p1_300_step_think_regional_binarized_rewards.jsonl \
#     --num_gpus 8 \
#     --processes_per_gpu 2 \
#     --binarize_score

# # 步骤 2: (可选) 格式化带有二值化分数的 region-based 输出结果
# python vlm_format_output.py \
#     --input_jsonl regional_rewards/images_comp_base_sub_0p1_300_step_think_regional_binarized_rewards.jsonl \
#     --output_jsonl images_comp/images_comp_base_sub_0p1_300_step_think_regional_binarized_rewards.jsonl