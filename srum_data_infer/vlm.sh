python srum_data_infer/vlm_multi_gpu.py \
    --model_path BAGEL-7B-MoT \
    --image_dir images_comp/bagel_base_7b_50_step_think \
    --input_jsonl images_comp/bagel_base_7b_50_step_think/output.jsonl \
    --output_jsonl regional_rewards/images_comp_base_7b_50_step_think_regional_rewards.jsonl \
    --num_gpus 8 \
    --processes_per_gpu 2 \

python srum_data_infer/vlm_format_output.py \
    --input_jsonl regional_rewards/images_comp_base_7b_50_step_think_regional_rewards.jsonl \
    --output_jsonl images_comp/images_comp_base_7b_50_step_think_regional_rewards.jsonl \
    --original_jsonl images_comp/bagel_base_7b_50_step_think/output.jsonl
