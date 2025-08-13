#!/bin/bash

# --- Slurm 資源申請配置 ---
#SBATCH -p efm_p              # 指定要提交到的資源分區 (partition)
#SBATCH -N 1                  # 使用 1 個節點 (node)
#SBATCH --gres=gpu:8          # 申請 8 張 GPU 卡
#SBATCH --quotatype=reserved      # 使用 auto 配額，靈活利用資源
#SBATCH -J Bagel-VLM          # 給您的作業取一個名字，方便識別
#SBATCH -o logs/vlm-%j.out    # 將標準輸出日誌保存到 logs/vlm-[作業ID].out
#SBATCH -e logs/vlm-%j.err    # 將錯誤日誌保存到 logs/vlm-[作業ID].err

# --- 您要執行的命令 ---

# 為了避免日誌文件因目錄不存在而創建失敗，先創建 logs 目錄
echo "創建日誌目錄..."
mkdir -p logs

# 激活您的 Conda 環境
echo "激活 Conda 環境..."
conda init
conda activate bagel

# 執行您的主程序
echo "開始執行 vlm.sh 腳本..."
# 在 sbatch 腳本中，推薦使用 srun 來啟動您的並行程序
srun bash scripts/train_reg.sh
srun bash tool/trans2hf.sh
srun bash wise_sub/1-Copy1.sh
echo "腳本執行完畢。"