#!/bin/bash

# 设置固定的模型模板路径
# TEMPLATE_MODEL="BAGEL-7B-MoT"

# # 使用 seq 命令生成 0.1 到 1.0 的序列，步长为 0.1
# # 对于每个生成的数字 i (例如 0.1, 0.2, 0.3...)
# for i in $(seq 0.1 0.1 1.0)
# do
#   # 1. 构建原始权重的路径
#   #    例如: results/checkpoint_reg_2e5_0.1
#   INPUT_CHECKPOINT_PATH="results/checkpoint_reg_2e5_${i}/0000974/"

#   # 2. 构建转换后模型的输出路径 (在原始路径后添加 _hf)
#   #    例如: results/checkpoint_reg_2e5_0.1_hf
#   OUTPUT_HF_PATH="results/hf_weights/checkpoint_reg_2e5_${i}_hf"

#   # 打印将要执行的命令，方便调试
#   echo "############################################################"
#   echo "### Processing: ${INPUT_CHECKPOINT_PATH}"
#   echo "### Output to:  ${OUTPUT_HF_PATH}"
#   echo "############################################################"

#   # 3. 执行 Python 转换脚本
#   python tool/trans2hf.py \
#     --training_checkpoint_path "${INPUT_CHECKPOINT_PATH}" \
#     --template_model_path "${TEMPLATE_MODEL}" \
#     --output_path "${OUTPUT_HF_PATH}"

# done

# echo "All checkpoints have been processed."

TEMPLATE_MODEL="BAGEL-7B-MoT"

# # 2. 构建原始权重的路径
# #    例如: results/checkpoint_reg_2e5_0.1/0000974/
# INPUT_CHECKPOINT_PATH="results/rft_cul_wise_con_0p5_2e5/0000395"

# # 3. 构建转换后模型的输出路径 (在原始路径后添加 _hf)
# #    例如: results/hf_weights/checkpoint_reg_2e5_0.1_hf
# OUTPUT_HF_PATH="results/hf_weights/rft_cul_wise_con_0p5_2e5_hf"

# # 打印将要执行的命令，方便调试
# echo "############################################################"
# echo "### Processing: ${INPUT_CHECKPOINT_PATH}"
# echo "### Output to:  ${OUTPUT_HF_PATH}"
# echo "############################################################"

# # 4. 执行 Python 转换脚本
# python tool/trans2hf.py \
#   --training_checkpoint_path "${INPUT_CHECKPOINT_PATH}" \
#   --template_model_path "${TEMPLATE_MODEL}" \
#   --output_path "${OUTPUT_HF_PATH}"

# echo "Checkpoint for weight has been processed."





# INPUT_CHECKPOINT_PATH="results/rft_ns_wise_con_0p5_2e5/0000296"

# # 3. 构建转换后模型的输出路径 (在原始路径后添加 _hf)
# #    例如: results/hf_weights/checkpoint_reg_2e5_0.1_hf
# OUTPUT_HF_PATH="results/hf_weights/rft_ns_wise_con_0p5_2e5_hf"

# # 打印将要执行的命令，方便调试
# echo "############################################################"
# echo "### Processing: ${INPUT_CHECKPOINT_PATH}"
# echo "### Output to:  ${OUTPUT_HF_PATH}"
# echo "############################################################"

# # 4. 执行 Python 转换脚本
# python tool/trans2hf.py \
#   --training_checkpoint_path "${INPUT_CHECKPOINT_PATH}" \
#   --template_model_path "${TEMPLATE_MODEL}" \
#   --output_path "${OUTPUT_HF_PATH}"

# echo "Checkpoint for weight has been processed."






INPUT_CHECKPOINT_PATH="results/rft_comp_0p1_reg_global_lam_0p5_2e5/0000690"

# 3. 构建转换后模型的输出路径 (在原始路径后添加 _hf)
#    例如: results/hf_weights/checkpoint_reg_2e5_0.1_hf
OUTPUT_HF_PATH="results/hf_weights/rft_comp_0p1_reg_global_lam_0p5_2e5_hf"

# 打印将要执行的命令，方便调试
echo "############################################################"
echo "### Processing: ${INPUT_CHECKPOINT_PATH}"
echo "### Output to:  ${OUTPUT_HF_PATH}"
echo "############################################################"

# 4. 执行 Python 转换脚本
python tool/trans2hf.py \
  --training_checkpoint_path "${INPUT_CHECKPOINT_PATH}" \
  --template_model_path "${TEMPLATE_MODEL}" \
  --output_path "${OUTPUT_HF_PATH}"

echo "Checkpoint for weight has been processed."