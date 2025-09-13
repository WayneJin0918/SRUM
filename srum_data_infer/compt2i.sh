#!/bin/bash

# Define the model path and the base output directory
MODEL_PATH="results/hf_weights/rft_comp_0p1_reg_global_lam_0p5_2e5_hf"
BASE_OUTPUT_DIR="images_comp/rft_comp_0p1_reg_global_lam_0p5_2e5_hf"
JSON_DIR="T2I-CompBench_dataset/sub_json"

# Ensure the base output directory exists
mkdir -p "$BASE_OUTPUT_DIR"

# Iterate over all json files in the specified directory
for metadata_file in "$JSON_DIR"/*.json; do
    # Extract the filename (without extension) from the file path
    filename=$(basename -- "$metadata_file")
    filename_no_ext="${filename%.*}_50_think"

    # Construct the output directory path
    output_dir="$BASE_OUTPUT_DIR/$filename_no_ext"

    # Print information about the file being processed
    echo "Processing $metadata_file"
    echo "Outputting to $output_dir"

    # Execute the Python script
    python t2i_multi_gpu.py \
        --model_path "$MODEL_PATH" \
        --output_dir "$output_dir" \
        --metadata_file "$metadata_file" \
        --num_timesteps "50" \
        --num_gpus 8 \
        --think \
        --procs_per_gpu 2
done

echo "All JSON files have been processed."

for metadata_file in "$JSON_DIR"/*.json; do
    # Extract the filename (without extension) from the file path
    filename=$(basename -- "$metadata_file")
    filename_no_ext="${filename%.*}_50_no_think"

    # Construct the output directory path
    output_dir="$BASE_OUTPUT_DIR/$filename_no_ext"

    # Print information about the file being processed
    echo "Processing $metadata_file"
    echo "Outputting to $output_dir"

    # Execute the Python script
    python t2i_multi_gpu.py \
        --model_path "$MODEL_PATH" \
        --output_dir "$output_dir" \
        --metadata_file "$metadata_file" \
        --num_timesteps "50" \
        --num_gpus 8 \
        --procs_per_gpu 2
done

echo "All JSON files have been processed."