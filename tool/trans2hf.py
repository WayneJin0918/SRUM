# -*- coding: utf-8 -*-
import os
import torch
import shutil
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import time
import concurrent.futures
import argparse # 导入 argparse 模块

# --- 全局配置 (可根据需要调整) ---

# 需要被转换为 bfloat16 的文件名
FILES_TO_CONVERT = {"model.safetensors", "ema.safetensors"}

# 并行处理时使用的最大进程数 (None 表示使用所有可用的 CPU核心)
MAX_WORKERS = None #可以设置为具体的数字，例如 4

# --- 工作函数 (用于并行处理) ---

def convert_file_to_bf16(file_info):
    """
    在单个进程中转换单个文件到 bfloat16 格式。
    这是一个独立的函数，以便于并行化。
    
    参数:
        file_info (tuple): 包含 (filename, source_folder, target_folder) 的元组。
        
    返回:
        str: 描述操作结果的消息。
    """
    filename, source_folder, target_folder = file_info
    source_path = os.path.join(source_folder, filename)
    target_path = os.path.join(target_folder, filename)
    
    try:
        # 加载权重文件到 CPU，避免占用 GPU 显存
        tensors = load_file(source_path, device="cpu")
        tensors_bf16 = {}
        
        # 使用 tqdm 显示单个文件内部张量的转换进度
        # leave=False 表示完成后进度条会消失
        item_iterator = tqdm(tensors.items(), desc=f"  -> 转换 '{filename}'", leave=False, position=1)
        for k, v in item_iterator:
            # 将张量转换为 bfloat16 类型
            tensors_bf16[k] = v.to(torch.bfloat16)
        
        # 保存转换后的文件
        save_file(tensors_bf16, target_path)
        return f"✅ [子进程] 成功转换并保存: '{target_path}'"
        
    except Exception as e:
        return f"❌ [子进程] 处理 '{filename}' 时发生错误: {e}"


# --- 主脚本 ---

def main(source_folder, master_model_folder, target_folder):
    """
    执行模型转换和权重补全全流程的主函数。
    
    参数:
        source_folder (str): 包含原始检查点的源文件夹路径。
        master_model_folder (str): 包含主模型配置、分词器等的文件夹路径。
        target_folder (str): 用于存放处理后文件的目标文件夹路径。
    """
    start_time = time.time()
    print("🚀 开始执行模型处理脚本 (并行加速版)...")
    print(f"源检查点文件夹: {source_folder}")
    print(f"源主模型文件夹: {master_model_folder}")
    print(f"目标文件夹: {target_folder}")
    print("-" * 60)

    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs(target_folder, exist_ok=True)
    
    # 根据传入的 master_model_folder 动态构建 master_ema_path
    master_ema_path = os.path.join(master_model_folder, "ema.safetensors")


    # --- 步骤 1: 复制配置文件和分词器等非权重文件 ---
    print("\n--- 步骤 1: 复制非权重文件 (如 config, tokenizer) ---")
    try:
        master_model_files = os.listdir(master_model_folder)
        
        # 使用 tqdm 显示文件复制进度
        copy_iterator = tqdm(master_model_files, desc="复制非权重文件")
        
        copied_count = 0
        for filename in copy_iterator:
            # 跳过所有权重文件，只复制其他类型文件
            if filename.endswith('ema.safetensors'):
                continue

            src_path = os.path.join(master_model_folder, filename)
            dst_path = os.path.join(target_folder, filename)
            
            # 确保我们正在复制的是文件，而不是子目录
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                copied_count += 1
        
        print(f"✅ 成功复制 {copied_count} 个非权重文件到 '{target_folder}'")

    except FileNotFoundError:
        print(f"❌ 错误: Master 模型文件夹不存在: {master_model_folder}")
    except Exception as e:
        print(f"❌ 在复制文件时发生错误: {e}")

    print("\n--- 步骤 1 完成 ---")
    print("-" * 60)


    # --- 步骤 2: 并行将指定文件转换为 bfloat16 格式 ---
    print("\n--- 步骤 2: 并行转换权重文件到 bfloat16 ---")
    
    try:
        all_source_files = os.listdir(source_folder)
    except FileNotFoundError:
        print(f"❌ 错误: 源检查点文件夹不存在: {source_folder}")
        return

    # 筛选出需要转换的文件
    files_to_process = [f for f in all_source_files if f in FILES_TO_CONVERT]
    
    if not files_to_process:
        print("🟡 在源检查点文件夹中未找到需要转换的权重文件。")
    else:
        tasks = [(filename, source_folder, target_folder) for filename in files_to_process]
        print(f"📨 将 {len(tasks)} 个权重文件转换任务提交到进程池...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(tqdm(executor.map(convert_file_to_bf16, tasks), total=len(tasks), desc="并行转换进度"))
        
        print("\n--- 并行转换结果 ---")
        for res in results:
            print(res)
        print("----------------------")

    print("\n--- 步骤 2 完成 ---")
    print("-" * 60)

    # --- 步骤 3: 为 model.safetensors 补全缺失的权重 ---
    print("\n--- 步骤 3: 为 model.safetensors 补全缺失的权重 ---")
    
    target_model_path = os.path.join(target_folder, "model.safetensors")
    
    if not os.path.exists(target_model_path):
        print(f"⚠️  警告: 目标模型 '{target_model_path}' 不存在。跳过权重补全步骤。")
    elif not os.path.exists(master_ema_path):
        print(f"⚠️  警告: 权重来源 (Master) '{master_ema_path}' 不存在。跳过权重补全步骤。")
    else:
        try:
            print("正在加载 Master 模型和目标模型...")
            master_ema_tensors = load_file(master_ema_path, device="cpu")
            target_model_tensors = load_file(target_model_path, device="cpu")

            master_keys = set(master_ema_tensors.keys())
            target_keys = set(target_model_tensors.keys())
            
            missing_keys = master_keys - target_keys

            if not missing_keys:
                print("✅ 'model.safetensors' 中的权重是完整的，无需补全。")
            else:
                print(f"🟡 发现 {len(missing_keys)} 个缺失的权重，现在开始补全...")
                
                merged_tensors = target_model_tensors.copy()

                key_iterator = tqdm(sorted(list(missing_keys)), desc="  -> 补全缺失的权重")
                for key in key_iterator:
                    merged_tensors[key] = master_ema_tensors[key].to(torch.bfloat16)
                
                print(f"💾 正在保存补全后的模型，总权重数: {len(merged_tensors)}...")
                save_file(merged_tensors, target_model_path)
                print(f"✅ 成功补全并保存至: '{target_model_path}'")

        except Exception as e:
            print(f"❌ 在权重补全过程中发生错误: {e}")

    print("\n--- 步骤 3 完成 ---")
    print("-" * 60)

    end_time = time.time()
    print(f"🎉 所有任务已完成，总耗时: {end_time - start_time:.2f} 秒。")
    
    flag_path = os.path.join(target_folder, "processing_complete.txt")
    with open(flag_path, "w", encoding="utf-8") as f:
        f.write(f"处理完成于: {time.ctime()}.\n")
        f.write("已复制所有非权重文件。\n")
        f.write(f"已将 {FILES_TO_CONVERT} 转换为 bfloat16 格式。\n")
        f.write("已使用 Master EMA 模型补全了 model.safetensors 的权重。\n")
    print(f"📄 已创建标记文件: '{flag_path}'")


if __name__ == "__main__":
    # --- 命令行参数解析 ---
    parser = argparse.ArgumentParser(description="将训练检查点转换为 Hugging Face 格式，并补全权重。")
    parser.add_argument(
        "--training_checkpoint_path",
        type=str,
        required=True,
        help="包含原始检查点（如 model.safetensors）的源文件夹路径。"
    )
    parser.add_argument(
        "--template_model_path",
        type=str,
        required=True,
        help="包含主模型配置、分词器和完整 ema.safetensors 的文件夹路径。"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="用于存放处理后文件的目标文件夹路径。"
    )
    args = parser.parse_args()

    # 在 Windows 或 macOS 上使用 'spawn' 或 'forkserver' 启动方式时，
    # 必须将主逻辑代码放在 if __name__ == "__main__": 块内。
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果启动方法已经设置，则忽略错误
        pass
        
    # 使用从命令行解析的参数调用主函数
    main(
        source_folder=args.training_checkpoint_path,
        master_model_folder=args.template_model_path,
        target_folder=args.output_path
    )