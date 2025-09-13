#!/bin/bash
export PYTHONPATH="/mnt/Bagel/"
# export WANDB_API_KEY="81dc34f0253dc006e90f97bfaf291beda833e155"
num_nodes=1        # Can be modified according to the actual cluster setup
node_rank=0        # Current node rank (needs to be configured according to the actual setup for multi-node/multi-machine)
master_addr="127.0.0.1"  # Master node address, localhost can be used for a single machine
master_port=29505      # Communication port, avoid conflicts with existing processes
# cd /mnt/Bagel
model_path="BAGEL-7B-MoT"

torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=8 \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file scripts/data/wise_reg.yaml \
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
  --save_every 295 \
  --total_steps 296 \
  --checkpoint_dir "results/st_lam_0p5_2e5" \
  --lambda_constraint 0.5 \