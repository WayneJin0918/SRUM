# #!/bin/bash

# # 设置一个从 0.1 到 1.0，步长为 0.1 的循环
# for weight in $(seq 0.1 0.1 1.0)
# do
#   # 打印当前正在处理的权重值，方便跟踪进度
#   echo "======================================================"
#   echo "               Processing weight: $weight              "
#   echo "======================================================"

#   # 根据当前的权重值，构建 model_path 和 output_dir 参数
#   MODEL_PATH="results/hf_weights/checkpoint_reg_2e5_${weight}_hf"
#   OUTPUT_DIR="images_wise/bagel_reg_reward_${weight}_7b_300_step_think"

#   # 执行 Python 命令
#   python t2i_multi_gpu.py \
#     --model_path "${MODEL_PATH}" \
#     --output_dir "${OUTPUT_DIR}" \
#     --metadata_file wise.json \
#     --num_timesteps "50" \
#     --num_gpus 8 \
#     --think

#   echo "Finished processing weight: $weight"
#   echo ""
# done

# echo "All weight processing tasks are complete."


# python  t2i_multi_gpu.py \
#     --model_path "BAGEL-7B-MoT" \
#     --output_dir images_comp/bagel_base_7b_300_step_no_think \
#     --metadata_file train_comp.json\
#     --num_timesteps "50"\
#     --num_gpus 8\
#     --procs_per_gpu 2\

# python  t2i_multi_gpu.py \
#     --model_path "BAGEL-7B-MoT" \
#     --output_dir images_comp/eval_data/bagel_base_7b_50_step_think \
#     --metadata_file T2I-CompBench_dataset/sub_json/3d_spatial.json\
#     --num_timesteps "50"\
#     --num_gpus 8\
#     # --think \
#     --procs_per_gpu 2\
    
# python  t2i_multi_gpu.py \
#     --model_path "results/hf_weights/checkpoint_reg_2e5_0.1_hf" \
#     --output_dir images_wise/bagel_base_7b_300_step_think \
#     --metadata_file wise.json\
#     --num_timesteps "300"\
#     --num_gpus 8\
#     --think \

# python  t2i_multi_gpu.py \
#     --model_path "BAGEL-7B-MoT" \
#     --output_dir images_wise/bagel_base_7b_300_step_think \
#     --metadata_file wise.json\
#     --num_timesteps "300"\
#     --num_gpus 8\
#     --think \
# python  t2i_multi_gpu.py \
#     --model_path "BAGEL-7B-MoT" \
#     --output_dir images_wise/bagel_base_7b_300_step_no_think \
#     --metadata_file wise.json\
#     --num_timesteps "250"\
#     --num_gpus 8\

# python  t2i_multi_gpu.py \
#     --model_path "BAGEL-7B-MoT" \
#     --output_dir images_wise/bagel_base_7b_300_step_think \
#     --metadata_file wise.json\
#     --num_timesteps "250"\
#     --num_gpus 8\
#     --think \

# python  t2i_multi_gpu.py \
#     --model_path "BAGEL-7B-MoT" \
#     --output_dir images_wise/bagel_base_7b_150_step_no_think \
#     --metadata_file wise.json\
#     --num_timesteps "150"\
#     --num_gpus 8\

# python  t2i_multi_gpu.py \
#     --model_path "BAGEL-7B-MoT" \
#     --output_dir images_wise/bagel_base_7b_150_step_think \
#     --metadata_file wise.json\
#     --num_timesteps "150"\
#     --num_gpus 8\
#     --think \

python  t2i_multi_gpu.py \
    --model_path "BAGEL-7B-MoT" \
    --output_dir images_wise/bagel_base_7b_50_step_think \
    --metadata_file wise.json\
    --num_timesteps "50"\
    --num_gpus 8\
    --think \

# python  t2i_multi_gpu.py \
#     --model_path "BAGEL-7B-MoT" \
#     --output_dir images_wise/bagel_base_7b_50_step_think \
#     --metadata_file wise.json\
#     --num_timesteps "50"\
#     --num_gpus 8\
#     --think \

    