# python vlm_multi_gpu.py \
#     --model_path BAGEL-7B-MoT \
#     --image_dir images_comp/bagel_base_7b_300_step_think \
#     --input_jsonl images_comp/bagel_base_7b_300_step_think/output_sub_0p1.jsonl \
#     --output_jsonl regional_rewards/images_comp_base_sub_0p1_7b_300_step_think_glo_and_regional_rewards.jsonl \
#     --num_gpus 8 \
#     --processes_per_gpu 2 \
#     --max_regions 10 \
#     --sam_model_path sam_vit_h_4b8939.pth \
#     --global_layout_reward

python vlm_format_output.py \
    --input_jsonl regional_rewards/images_comp_base_sub_0p1_7b_300_step_think_glo_and_regional_rewards.jsonl \
    --output_jsonl images_comp/images_comp_base_sub_0p1_7b_300_step_think_glo_and_regional_rewards.jsonl \
    --original_jsonl images_comp/bagel_base_7b_300_step_think/output_sub_0p1.jsonl