# #!/bin/bash

# # --- 变量设置 ---
# # 您希望使用的 GPU 数量
# GPUS=8
# # BAGEL 模型检查点路径
# MODEL_PATH="/mnt/data/checkpoints/BAGEL-7B-MoT"
# # 输出图片和思考文本的目录
# OUTPUT_DIR="/mnt/data/nyw/Bagel/output"
# # 包含提示的 JSON 元数据文件路径
# METADATA_FILE="/mnt/data/nyw/Bagel/Binomial_operation_1000.json"
# # 您的 Python 脚本路径（假设您将上述代码保存为 t2i.py）
# PYTHON_SCRIPT="/mnt/data/nyw/Bagel/t2i.py"

# # --- 准备环境变量 ---
# echo "--- Preparing environment variables ---"
# export PYTHONPATH=$PWD:$PYTHONPATH
# echo "PYTHONPATH=$PYTHONPATH"

# # --- 开始并行图像生成 ---
# echo "--- Starting parallel image generation ---"
# echo "Running command: torchrun \\"
# echo "    --nnodes=1 \\"
# echo "    --node_rank=0 \\"
# echo "    --nproc_per_node=\$GPUS \\"
# echo "    --master_addr=127.0.0.1 \\"
# echo "    --master_port=12345 \\"
# echo "    \"\$PYTHON_SCRIPT\" \\"
# echo "    --model_path \"\$MODEL_PATH\" \\"
# echo "    --output_dir \"\$OUTPUT_DIR\" \\"
# echo "    --metadata_file \"\$METADATA_FILE\" \\"
# echo "    --think" # 根据您的需求决定是否启用 --think 模式

# torchrun \
#     --nnodes=1 \
#     --node_rank=0 \
#     --nproc_per_node=$GPUS \
#     --master_addr="127.0.0.1" \
#     --master_port="12345" \
#     "$PYTHON_SCRIPT" \
#     --model_path "$MODEL_PATH" \
#     --output_dir "$OUTPUT_DIR" \
#     --metadata_file "$METADATA_FILE" \
#     --think # 如果不需要思考模式，请删除此行或将其改为 --no-think (如果脚本支持)

# echo "--- Parallel image generation completed ---"

# torchrun --nproc_per_node=8 /mnt/data/nyw/Bagel/t2i_multi_gpu.py \
#     --model_path /mnt/data/checkpoints/BAGEL-7B-MoT \
#     --output_dir ./generated_images \
#     --metadata_file /mnt/data/nyw/Bagel/Binomial_operation_1000.json
# 8张GPU并行处理1000个prompt


# torchrun --nproc_per_node=8 t2i_ddp.py \
#     --model_path /path/to/your/BAGEL-7B-MoT \
#     --output_dir /path/to/save/images \
#     --metadata_file /path/to/your/metadatas.json \
#     --think # 如果需要 think 模式，则添加此项
#     # --overwrite # 如果需要覆盖已存在文件，则添加此项
# cd /mnt/data/nyw/Bagel
# python  t2i_multi_gpu.py \
#     --model_path /yuchang/lsy_jwy/Bagel/BAGEL-7B-MoT \
#     --output_dir ./generated_images_no_think_wise_100_step \
#     --metadata_file /yuchang/lsy_jwy/Bagel/WhyUni/wise.json\
#     --num_timesteps "100"\
#     --num_gpus 4\
#     # --think

python  t2i_multi_gpu.py \
    --model_path /yuchang/lsy_jwy/Bagel/BAGEL-7B-MoT \
    --output_dir ./generated_images_think_wise_200_step \
    --metadata_file /yuchang/lsy_jwy/Bagel/WhyUni/wise.json\
    --num_timesteps "200"\
    --num_gpus 4\
    --think