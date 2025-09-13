#!/bin/bash
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