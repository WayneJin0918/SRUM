# filename: create_subset.py

import os
import random
from tqdm import tqdm

def create_random_subset(input_path, output_path, sample_rate=0.1):
    """
    从一个大的JSONL文件中逐行随机采样，创建一个子集。

    Args:
        input_path (str): 原始 .jsonl 文件的路径。
        output_path (str): 输出的子集 .jsonl 文件的路径。
        sample_rate (float): 采样率 (例如 0.1 表示 10%)。
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    try:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            print(f"开始处理文件: {input_path}")
            print(f"采样率为: {sample_rate * 100:.1f}%")
    
            lines_written = 0
            
            # --- MODIFICATION START ---
            # 1. 将 tqdm 实例赋值给一个变量，例如 progress_bar
            progress_bar = tqdm(f_in, desc="正在处理行", unit="行")
            
            # 2. 遍历这个变量
            for line in progress_bar:
                if random.random() < sample_rate:
                    f_out.write(line)
                    lines_written += 1
            
            # 3. 循环结束后，从该变量的 .n 属性获取总行数
            total_lines_read = progress_bar.n 
            # --- MODIFICATION END ---
            
            print("\n处理完成！")
            print(f"总共读取行数: {total_lines_read}")
            print(f"成功写入行数: {lines_written}")
            print(f"子集文件已保存至: {output_path}")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到 -> {input_path}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    # --- 文件路径配置 ---
    base_dir = "images_comp/bagel_base_7b_300_step_think"
    input_file = os.path.join(base_dir, "output.jsonl")
    output_file = os.path.join(base_dir, "output_sub_0p1.jsonl")

    # --- 采样率 ---
    # 1/10 = 0.1
    sampling_rate = 0.15

    # --- 执行函数 ---
    create_random_subset(input_file, output_file, sampling_rate)