#!/bin/bash
INPUT_CHECKPOINT_PATH="results/rft_comp_0p1_reg_global_lam_0p5_2e5/0000690"

# 3. Construct the output path for the converted model (append _hf to the original path)
#    Example: results/hf_weights/checkpoint_reg_2e5_0.1_hf
OUTPUT_HF_PATH="results/hf_weights/rft_comp_0p1_reg_global_lam_0p5_2e5_hf"

# Print the command that will be executed, for easy debugging
echo "############################################################"
echo "### Processing: ${INPUT_CHECKPOINT_PATH}"
echo "### Output to:  ${OUTPUT_HF_PATH}"
echo "############################################################"

# 4. Execute the Python conversion script
python tool/trans2hf.py \
  --training_checkpoint_path "${INPUT_CHECKPOINT_PATH}" \
  --template_model_path "${TEMPLATE_MODEL}" \
  --output_path "${OUTPUT_HF_PATH}"

echo "Checkpoint for weight has been processed."