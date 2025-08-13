#!/bin/bash

# 定义模型路径和基础输出目录
MODEL_PATH="results/hf_weights/rft_comp_0p1_reg_global_lam_0p5_2e5_hf"
BASE_OUTPUT_DIR="images_comp/rft_comp_0p1_reg_global_lam_0p5_2e5_hf"
JSON_DIR="T2I-CompBench_dataset/sub_json"

# 确保基础输出目录存在
mkdir -p "$BASE_OUTPUT_DIR"

# 遍历指定目录下的所有json文件
for metadata_file in "$JSON_DIR"/*.json; do
    # 从文件路径中提取文件名（不含扩展名）
    filename=$(basename -- "$metadata_file")
    filename_no_ext="${filename%.*}_50_think"

    # 构建输出目录路径
    output_dir="$BASE_OUTPUT_DIR/$filename_no_ext"

    # 打印正在处理的文件信息
    echo "Processing $metadata_file"
    echo "Outputting to $output_dir"

    # 执行Python脚本
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
    # 从文件路径中提取文件名（不含扩展名）
    filename=$(basename -- "$metadata_file")
    filename_no_ext="${filename%.*}_50_no_think"

    # 构建输出目录路径
    output_dir="$BASE_OUTPUT_DIR/$filename_no_ext"

    # 打印正在处理的文件信息
    echo "Processing $metadata_file"
    echo "Outputting to $output_dir"

    # 执行Python脚本
    python t2i_multi_gpu.py \
        --model_path "$MODEL_PATH" \
        --output_dir "$output_dir" \
        --metadata_file "$metadata_file" \
        --num_timesteps "50" \
        --num_gpus 8 \
        --procs_per_gpu 2
done

echo "All JSON files have been processed."