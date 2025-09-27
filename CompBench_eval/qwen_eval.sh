#!/bin/bash

# =================================================================
# Configuration Area: Please modify these variables for your environment
# =================================================================

# 1. Base directory containing all category folders
BASE_RESULTS_DIR="SRUM/comp_eval/rft_comp_image"

# 2. Output directory for evaluation results
OUTPUT_DIR="SRUM/comp_eval/rft_comp_eval"

# 3. Model and other parameters
MODEL_ID="checkpoints/Qwen2.5-VL-72B-Instruct"
BATCH_SIZE=8
NUM_WORKERS=2

# 4. [New] Filename for the summary results
SUMMARY_CSV="$OUTPUT_DIR/summary_results.csv"


# =================================================================
# Script Body: You usually don't need to modify the section below
# =================================================================

# Define the model variants to be evaluated
VARIANTS=("think" "no think")

# Define all CompBench categories to be evaluated
BASE_CATEGORIES=("color" "shape" "texture" "spatial" "non-spatial" "numeracy" "3d spatial")
COMPLEX_PREFIXES=("complex" "complex action" "complex spatial")

# Check and create the output directory
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Output directory does not exist, creating: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Check if the base results directory exists
if [ ! -d "$BASE_RESULTS_DIR" ]; then
    echo "Error: Base results directory does not exist: $BASE_RESULTS_DIR"
    exit 1
fi

# Record the start time
START_TIME=$(date +%s)
echo "Automated comparison evaluation started..."
echo "Model: $MODEL_ID"
echo "------------------------------------------------"

# [New] Initialize the summary CSV file with a header
echo "Variant,Category,Mean_Score" > "$SUMMARY_CSV"

# Outer loop: iterate through each model variant
for VARIANT in "${VARIANTS[@]}"; do
    echo ""
    echo "#################################################"
    echo "### Processing model variant: $VARIANT"
    echo "#################################################"

    ALL_CATEGORIES=("${BASE_CATEGORIES[@]}" "${COMPLEX_PREFIXES[@]}")

    # Inner loop: iterate through each evaluation category
    for CATEGORY in "${ALL_CATEGORIES[@]}"; do
        CATEGORY_FILENAME_FRIENDLY=${CATEGORY// /_}
        VARIANT_FILENAME_FRIENDLY=${VARIANT// /_}
        
        # Construct the source folder name and path
        SOURCE_FOLDER_NAME="${CATEGORY_FILENAME_FRIENDLY}_50_${VARIANT_FILENAME_FRIENDLY}"
        IMAGE_PATH="$BASE_RESULTS_DIR/$SOURCE_FOLDER_NAME"
        
        # Check if the image directory exists
        if [ ! -d "$IMAGE_PATH" ]; then
            echo "Warning: Image directory not found, skipping: $IMAGE_PATH"
            continue
        fi

        echo ""
        echo "================================================="
        echo "Evaluating Category: '$CATEGORY' | Variant: '$VARIANT'"
        echo "================================================="

        # Define output and log file paths for the current run
        OUTPUT_CSV="$OUTPUT_DIR/${CATEGORY_FILENAME_FRIENDLY}_${VARIANT_FILENAME_FRIENDLY}_results.csv"
        LOG_FILE="$OUTPUT_DIR/${CATEGORY_FILENAME_FRIENDLY}_${VARIANT_FILENAME_FRIENDLY}_run.log"

        # Handle special category argument for the Python script
        PYTHON_CATEGORY_ARG=$CATEGORY
        if [[ $CATEGORY == "complex action" || $CATEGORY == "complex spatial" ]]; then
            PYTHON_CATEGORY_ARG="complex"
        fi

        # Execute the Python evaluation script
        python comp_eval/qwen_comp_eval.py \
            --image_path "$IMAGE_PATH" \
            --category "$PYTHON_CATEGORY_ARG" \
            --output_csv "$OUTPUT_CSV" \
            --log_file "$LOG_FILE" \
            --model_id "$MODEL_ID" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS"

        echo "Evaluation complete. Results saved in $OUTPUT_CSV"
        echo "Log saved in $LOG_FILE"

        # [New] Calculate the mean score from the output CSV and append to the summary file
        # This assumes the result CSV has a header and the score is in the second column.
        # Format example: image_path,score
        if [ -f "$OUTPUT_CSV" ]; then
            MEAN_SCORE=$(awk -F, 'NR > 1 { total += $2; count++ } END { if (count > 0) print total/count; else print 0 }' "$OUTPUT_CSV")
            echo "Calculated Mean Score for '$CATEGORY' ($VARIANT): $MEAN_SCORE"
            echo "$VARIANT,$CATEGORY,$MEAN_SCORE" >> "$SUMMARY_CSV"
        else
            echo "Warning: Output file $OUTPUT_CSV not found. Cannot calculate mean score."
            echo "$VARIANT,$CATEGORY,0" >> "$SUMMARY_CSV"
        fi
    done
done

# [New] Calculate the overall average from the summary file
OVERALL_AVERAGE=$(awk -F, 'NR > 1 { total += $3; count++ } END { if (count > 0) print total/count; else print 0 }' "$SUMMARY_CSV")
echo "" >> "$SUMMARY_CSV" # Add a blank line for readability
echo "Overall_Average,,$OVERALL_AVERAGE" >> "$SUMMARY_CSV"


# Record the end time and calculate the total duration
END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))
echo ""
echo "------------------------------------------------"
echo "All evaluation tasks have been completed!"
echo "Total execution time: $TOTAL_SECONDS seconds."
echo "All individual result files are saved in: $OUTPUT_DIR"
echo "A summary of mean scores is available at: $SUMMARY_CSV"
echo "================================================="