#!/bin/bash

# =================================================================
# Configuration Area: Please modify these variables for your environment
# =================================================================

# 1. Number of GPUs to use for Tensor Parallelism
NUM_GPUS=8

# 2. Base directory containing all category folders
BASE_RESULTS_DIR="SRUM/comp_eval/rft_comp_2_round_hf_image"

# 3. Output directory for evaluation results
OUTPUT_DIR="SRUM/comp_eval/rft_comp_2_round_hf_eval"

# 4. Model ID or path
MODEL_ID="checkpoints/Qwen2.5-VL-72B-Instruct"
# Or you can change to MODEL_ID="checkpoints/Qwen2.5-VL-32B-Instruct" duo to memory limitations

BATCH_SIZE=1

# 5. Filename for the summary results
SUMMARY_CSV="$OUTPUT_DIR/summary_results.csv"


# =================================================================
# Script Body: You usually don't need to modify the section below
# =================================================================

# Ensure we are using the correct python script name
PYTHON_SCRIPT="comp_eval/qwen_eval.py"

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
echo "Automated comparison evaluation started with DeepSpeed..."
echo "Model: $MODEL_ID"
echo "Using $NUM_GPUS GPUs for Tensor Parallelism."
echo "------------------------------------------------"

# Initialize the summary CSV file with a header
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

        # Execute the Python evaluation script using the Deepspeed launcher
        deepspeed --include="localhost:0,1,2,3,4,5,6,7" "$PYTHON_SCRIPT" \
            --image_path "$IMAGE_PATH" \
            --category "$PYTHON_CATEGORY_ARG" \
            --output_csv "$OUTPUT_CSV" \
            --log_file "$LOG_FILE" \
            --model_id "$MODEL_ID" \
            --batch_size "$BATCH_SIZE"

        echo "Evaluation complete. Results saved in $OUTPUT_CSV"
        echo "Log saved in $LOG_FILE"

        # Calculate the mean score from the output CSV and append to the summary file
        # The Python script now handles summary generation internally, but we can keep this as a fallback
        # Let's use the summary from the python script's log instead.
        if [ -f "$OUTPUT_CSV" ]; then
            # We can parse the final score from the log file, which is more robust.
            # The python script now logs the final average score.
            MEAN_SCORE=$(grep "Average Score for this run:" "$LOG_FILE" | awk '{print $NF}')
            if [ -z "$MEAN_SCORE" ]; then
                # Fallback to awk on the CSV if grep fails
                MEAN_SCORE=$(awk -F, 'BEGIN{total=0; count=0} /^[^#]/ && NR > 1 { if ($4 >= 0) { total+=$4; count++ } } END{if(count>0) printf "%.4f", total/count; else print "0.0000"}' "$OUTPUT_CSV")
            fi
            echo "Calculated Mean Score for '$CATEGORY' ($VARIANT): $MEAN_SCORE"
            echo "$VARIANT,$CATEGORY,$MEAN_SCORE" >> "$SUMMARY_CSV"
        else
            echo "Warning: Output file $OUTPUT_CSV not found. Cannot calculate mean score."
            echo "$VARIANT,$CATEGORY,0" >> "$SUMMARY_CSV"
        fi
    done
done

# Calculate the overall average from the summary file
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