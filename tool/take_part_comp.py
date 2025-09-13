import json
import os

def split_json_by_category(input_filename='val_comp.json', output_dir='T2I-CompBench_dataset/sub_json'):
    """
    Reads a JSON file, groups items by the 'Category' field, and writes each group to a separate JSON file.

    Args:
        input_filename (str): The name of the input JSON file.
        output_dir (str): The name of the directory to save the output JSON files.
    """
    try:
        # Open and load the source JSON file
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found. Please ensure the file path is correct.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not parse '{input_filename}'. Please ensure it is a valid JSON file.")
        return

    # Create a dictionary to hold items grouped by category
    categorized_data = {}

    # Iterate over each item in the data and group by 'Category'
    for item in data:
        # Get the category from the item, default to 'Uncategorized' if not found
        category = item.get('Category', 'Uncategorized')
        
        if category not in categorized_data:
            categorized_data[category] = []
        
        # Append the item to the corresponding category list
        categorized_data[category].append(item)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write each category's data to a new JSON file inside the output directory
    for category, items in categorized_data.items():
        # Sanitize the category name to create a valid filename
        safe_filename = "".join(c for c in category if c.isalnum() or c in (' ', '_')).rstrip()
        output_filename = os.path.join(output_dir, f'{safe_filename}.json')
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            # Use indent for pretty-printing the JSON
            # ensure_ascii=False allows for proper handling of non-ASCII characters
            json.dump(items, f, indent=4, ensure_ascii=False)
        
        print(f"File created: {output_filename}")

    print(f"\nProcessing complete! All files have been saved in the '{output_dir}' directory.")

# --- Run the code ---
if __name__ == '__main__':
    # You can change the input filename here if needed
    # Example: split_json_by_category('Bagel/val_comp.json')
    split_json_by_category('val_comp.json')