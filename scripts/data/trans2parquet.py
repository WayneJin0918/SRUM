import json
import os
import pandas as pd
from PIL import Image
import io
from tqdm import tqdm # 導入 tqdm

def create_parquet_from_jsonl(jsonl_path, image_dir, output_parquet_path):
    """
    從 JSONL 文件和圖片檔案夾創建 Parquet 文件。

    Args:
        jsonl_path (str): 輸入的 JSONL 文件路徑。
        image_dir (str): 包含圖片的目錄路徑。
        output_parquet_path (str): 輸出的 Parquet 文件路徑。
    """
    # 用於儲存所有處理好的數據行
    data_for_parquet = []

    # --- 新增部分：計算總行數以供 tqdm 使用 ---
    print("正在計算文件總行數...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        num_lines = sum(1 for _ in f)
    print(f"文件總行數: {num_lines}")
    # -----------------------------------------

    print(f"開始讀取與處理 JSONL 文件: {jsonl_path}")

    # 逐行讀取 JSONL 文件
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        # --- 修改部分：使用 tqdm 包裹迴圈 ---
        # tqdm 會自動處理進度條的顯示與更新
        for i, line in enumerate(tqdm(f, total=num_lines, desc="處理進度", unit="行")):
            try:
                item = json.loads(line)

                # 1. 提取所需資訊
                prompt_text = item.get('original_prompt')
                image_filename = item.get('image_file')

                if not prompt_text or not image_filename:
                    # 為了不讓警告訊息洗掉進度條，這裡可以選擇性地關閉
                    # print(f"\n警告: 第 {i+1} 行缺少 'original_prompt' 或 'image_file'，已跳過。")
                    continue
                
                # 2. 構建完整的圖片路徑
                image_path = os.path.join(image_dir, image_filename)

                if not os.path.exists(image_path):
                    # print(f"\n警告: 找不到圖片文件 {image_path}，已跳過。")
                    continue

                # 3. 讀取圖片並轉換為二進位 (bytes)
                with Image.open(image_path) as img:
                    # 將圖片轉換為 RGB 格式，避免 RGBA 或 P 模式等問題
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    byte_arr = io.BytesIO()
                    img.save(byte_arr, format='JPEG') # 統一儲存為 JPEG 格式以節省空間
                    image_bytes = byte_arr.getvalue()

                # 4. 將 prompt 格式化為 JSON 字串
                caption_dict = {"caption": prompt_text}
                caption_json_str = json.dumps(caption_dict, ensure_ascii=False)

                # 5. 將處理好的數據加入列表
                data_for_parquet.append({
                    'image': image_bytes,       # 圖片二進位數據
                    'captions': caption_json_str # 符合格式的 JSON 字串
                })

            except Exception as e:
                # 輸出錯誤時，在訊息前加上換行符，避免與進度條同行
                print(f"\n處理第 {i+1} 行時發生錯誤: {e}")

    if not data_for_parquet:
        print("沒有成功處理任何數據，無法生成 Parquet 文件。")
        return

    print(f"\n共處理了 {len(data_for_parquet)} 條有效的數據。")
    print(f"正在寫入 Parquet 文件至: {output_parquet_path}")

    # 使用 Pandas DataFrame 寫入 Parquet 文件
    df = pd.DataFrame(data_for_parquet)
    df.to_parquet(output_parquet_path, engine='pyarrow', compression='snappy')

    print("Parquet 文件已成功創建！ ✨")


if __name__ == '__main__':
    # --- 請在此處配置您的路徑 ---
    base_dir = "images_wise/bagel_base_7b_300_step_think"
    
    # 輸入文件路徑
    jsonl_file_path = os.path.join(base_dir, "output.jsonl")
    
    # 圖片所在的目錄
    images_directory = base_dir
    
    # 輸出的 Parquet 文件路徑
    # 確保輸出的目錄存在
    output_dir = "images_wise/bagel_base_7b_300_step_think_parquet"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "output_dataset.parquet")
    
    # 執行轉換
    create_parquet_from_jsonl(jsonl_file_path, images_directory, output_file_path)