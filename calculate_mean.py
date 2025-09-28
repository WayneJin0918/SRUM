import pandas as pd
import sys
import os

def calculate_mean_score(file_path):
    """
    Calculates the mean of the 'score' column from a given CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        float: The mean score, or 0.0 if an error occurs or the file is empty.
    """
    # Check if the file exists and is not empty
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        print(f"Warning: File not found or is empty: {file_path}", file=sys.stderr)
        return 0.0

    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Check if the 'score' column exists
        if 'score' not in df.columns:
            print(f"Error: 'score' column not found in {file_path}.", file=sys.stderr)
            return 0.0

        # Convert the 'score' column to a numeric type.
        # 'coerce' will turn any non-numeric values into Not-a-Number (NaN).
        scores = pd.to_numeric(df['score'], errors='coerce')

        # Drop any rows where the score could not be converted to a number
        valid_scores = scores.dropna()

        # Calculate the mean if there are any valid scores
        if not valid_scores.empty:
            mean_score = valid_scores.mean()
            return mean_score
        else:
            print(f"Warning: No valid scores found in {file_path}.", file=sys.stderr)
            return 0.0

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}", file=sys.stderr)
        return 0.0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calculate_mean.py <path_to_csv_file>", file=sys.stderr)
        sys.exit(1)

    csv_file_path = sys.argv[1]
    mean_value = calculate_mean_score(csv_file_path)
    # Print the result formatted to 4 decimal places, which the bash script will capture
    print(f"{mean_value:.4f}")