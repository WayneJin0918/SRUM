import json
import os
import pandas as pd
from PIL import Image
import io
from tqdm import tqdm # Import tqdm

def create_parquet_from_jsonl(jsonl_path, image_dir, output_parquet_path):
    """
    Creates a Parquet file from a JSONL file and an image folder.

    Args:
        jsonl_path (str): Path to the input JSONL file.
        image_dir (str): Path to the directory containing the images.
        output_parquet_path (str): Path for the output Parquet file.
    """
    # Used to store all processed data rows
    data_for_parquet = []

    # --- New part: Calculate the total number of lines for tqdm ---
    print("Calculating the total number of lines in the file...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        num_lines = sum(1 for _ in f)
    print(f"Total lines in file: {num_lines}")
    # -----------------------------------------

    print(f"Starting to read and process the JSONL file: {jsonl_path}")

    # Read the JSONL file line by line
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        # --- Modified part: Wrap the loop with tqdm ---
        # tqdm will automatically handle the display and update of the progress bar
        for i, line in enumerate(tqdm(f, total=num_lines, desc="Processing progress", unit="lines")):
            try:
                item = json.loads(line)

                # 1. Extract required information
                prompt_text = item.get('original_prompt')
                image_filename = item.get('image_file')

                if not prompt_text or not image_filename:
                    # To prevent warning messages from overwriting the progress bar, this can be optionally disabled
                    # print(f"\nWarning: Line {i+1} is missing 'original_prompt' or 'image_file', skipping.")
                    continue
                
                # 2. Construct the full image path
                image_path = os.path.join(image_dir, image_filename)

                if not os.path.exists(image_path):
                    # print(f"\nWarning: Image file not found at {image_path}, skipping.")
                    continue

                # 3. Read the image and convert it to binary (bytes)
                with Image.open(image_path) as img:
                    # Convert the image to RGB format to avoid issues with modes like RGBA or P
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    byte_arr = io.BytesIO()
                    img.save(byte_arr, format='JPEG') # Save uniformly as JPEG format to save space
                    image_bytes = byte_arr.getvalue()

                # 4. Format the prompt into a JSON string
                caption_dict = {"caption": prompt_text}
                caption_json_str = json.dumps(caption_dict, ensure_ascii=False)

                # 5. Add the processed data to the list
                data_for_parquet.append({
                    'image': image_bytes,      # Image binary data
                    'captions': caption_json_str # Formatted JSON string
                })

            except Exception as e:
                # When printing an error, add a newline character at the beginning to avoid being on the same line as the progress bar
                print(f"\nAn error occurred while processing line {i+1}: {e}")

    if not data_for_parquet:
        print("No data was processed successfully. Cannot generate Parquet file.")
        return

    print(f"\nProcessed a total of {len(data_for_parquet)} valid data entries.")
    print(f"Writing Parquet file to: {output_parquet_path}")

    # Use a Pandas DataFrame to write the Parquet file
    df = pd.DataFrame(data_for_parquet)
    df.to_parquet(output_parquet_path, engine='pyarrow', compression='snappy')

    print("Parquet file created successfully! ✨")


if __name__ == '__main__':
    # --- Please configure your paths here ---
    base_dir = "images_comp/bagel_base_7b_300_step_think"
    
    # Input file path
    jsonl_file_path = os.path.join(base_dir, "output.jsonl")
    
    # Directory where the images are located
    images_directory = base_dir
    
    # Output Parquet file path
    # Ensure the output directory exists
    output_dir = "images_comp/bagel_base_7b_300_step_think_parquet"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "output_dataset.parquet")
    
    # Execute the conversion
    create_parquet_from_jsonl(jsonl_file_path, images_directory, output_file_path)