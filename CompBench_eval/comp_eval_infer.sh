#!/bin/bash

# =================================================================
# Configuration Area: Modify these variables for your environment
# =================================================================

# 1. Path to the fine-tuned model weights
MODEL_PATH="SRUM/results/hf_weights/rft_comp"

# 2. Base directory where category subfolders will be created for the output images
#    This path is structured to align with evaluation script requirements.
BASE_OUTPUT_DIR="SRUM/comp_eval/rft_comp_image"

# 3. Path to the input metadata file for validation
METADATA_FILE="SRUM/val_comp.json"

# 4. Directory to store log files for each category's generation run
LOG_DIR="SRUM/logs/val_generation_logs"

# 5. Inference parameters
NUM_TIMESTEPS="50"
NUM_GPUS=8
PROCS_PER_GPU=1 # Number of processes to run on each GPU

# =================================================================
# Script Body: No modification needed below this line
# =================================================================

# --- Environment Setup ---
export NCCL_DEBUG=INFO
export NCCL_IB_TC=106
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_CROSS_NIC=0
mkdir -p ${LOG_DIR}

# --- Define Categories from the metadata file ---
# These must exactly match the "Category" field in val_comp.json
CATEGORIES=(
    "3d_spatial"
    "color"
    "complex"
    "complex_action"
    "complex_spatial"
    "non-spatial"
    "numeracy"
    "shape"
    "texture"
    "spatial"
)

# Record the start time
START_TIME=$(date +%s)
echo "Starting batch image generation for all validation categories..."
echo "Using metadata from: ${METADATA_FILE}"
echo "-------------------------------------------------"

# --- Main Generation Loop ---
for CATEGORY in "${CATEGORIES[@]}"; do
    echo ""
    echo "#################################################"
    echo "### Processing Category: ${CATEGORY}"
    echo "#################################################"

    # Create a filename-friendly version of the category name (e.g., "complex action" -> "complex_action")
    CATEGORY_FILENAME_FRIENDLY=$(echo "${CATEGORY}" | tr ' ' '_')
    
    # Create a temporary metadata file for the current category in a temporary directory
    TEMP_METADATA_FILE="/tmp/metadata_${CATEGORY_FILENAME_FRIENDLY}.json"
    
    # Define the output directory for this specific category, aligning with evaluation script naming
    # Format: <category>_<timesteps>_<mode>
    CATEGORY_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${CATEGORY_FILENAME_FRIENDLY}_${NUM_TIMESTEPS}_think"
    mkdir -p "${CATEGORY_OUTPUT_DIR}"

    # Use 'jq' to filter val_comp.json and create a temporary file for the current category
    echo "Creating temporary metadata file for '${CATEGORY}' at ${TEMP_METADATA_FILE}"
    jq --arg cat "$CATEGORY" '[.[] | select(.Category == $cat)]' "$METADATA_FILE" > "$TEMP_METADATA_FILE"

    # Verify that the temporary file was created and is not empty
    if [ ! -s "$TEMP_METADATA_FILE" ]; then
        echo "Warning: No prompts found for category '${CATEGORY}'. Skipping."
        continue
    fi

    # Define the log file for the current run
    LOG_FILE="${LOG_DIR}/generation_${CATEGORY_FILENAME_FRIENDLY}_$(date +%Y%m%d_%H%M%S).log"
    echo "Starting generation for '${CATEGORY}'. Output will be saved to: ${CATEGORY_OUTPUT_DIR}"
    echo "Log file for this run: ${LOG_FILE}"

    # Execute the multi-GPU generation script
    python SRUM/srum_data_infer/t2i_multi_gpu.py \
        --model_path "${MODEL_PATH}" \
        --output_dir "${CATEGORY_OUTPUT_DIR}" \
        --metadata_file "${TEMP_METADATA_FILE}" \
        --num_timesteps "${NUM_TIMESTEPS}" \
        --num_gpus ${NUM_GPUS} \
        --procs_per_gpu ${PROCS_PER_GPU} \
        --think \
        --overwrite 2>&1 | tee "${LOG_FILE}"

    echo "Finished processing category: ${CATEGORY}"
done

# --- Cleanup ---
echo "#################################################"
echo "Cleaning up temporary metadata files..."
rm /tmp/metadata_*.json

# --- Completion Summary ---
END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))
echo "-------------------------------------------------"
echo "All categories processed successfully!"
echo "Total execution time: ${TOTAL_SECONDS} seconds."
echo "Generated images are organized in subdirectories inside: ${BASE_OUTPUT_DIR}"
echo "================================================="