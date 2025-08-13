#!/bin/bash
export PYTHONPATH="/mnt/petrelfs/jinweiyang/Bagel/"
# 节点数
num_nodes=1         # 可根据集群实际情况修改
node_rank=0         # 当前节点编号（多节点多机需要按实际配置）
# master_addr="127.0.0.1"  # 主节点地址，单机可用本地
master_port=29505        # 通信端口，避免与已有进程冲突
# cd /mnt/data/nyw/Bagel
# 模型路径
model_path="BAGEL-7B-MoT"
# export PYTHONPATH=/mnt/data/nyw/Bagel:$PYTHONPATH
# export WANDB_API_KEY="81dc34f0253dc006e90f97bfaf291beda833e155"

torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=8 \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file scripts/data/sft_comp_0p5.yaml \
  --model_path $model_path \
  --visual_gen True \
  --visual_und False \
  --freeze_llm False \
  --freeze_vit True \
  --freeze_vae True \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from $model_path \
  --finetune_from_hf True \
  --cpu_offload False \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --num_worker 1 \
  --expected_num_tokens 8192 \
  --max_num_tokens 9216 \
  --max_num_tokens_per_sample 8192 \
  --save_every 3117 \
  --total_steps 3118 \
  --checkpoint_dir "results/comp_0p5_sft" \