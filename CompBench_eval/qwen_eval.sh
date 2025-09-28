#!/bin/bash

# =================================================================
# =================================================================
# Configuration Area: Please modify these variables for your environment
# =================================================================

# 1. Number of GPUs to use for parallel evaluation
NUM_GPUS=8

# 2. Base directory containing all category folders
BASE_RESULTS_DIR="/SRUM/comp_eval/rft_comp_1_round_hf_image"

# 3. Output directory for evaluation results
OUTPUT_DIR="/SRUM/comp_eval/rft_comp_1_round_hf_eval"

# 4. Model ID or path
MODEL_ID="/checkpoints/Qwen2.5-VL-32B-Instruct"
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
echo "Automated evaluation started in parallel mode..."
echo "Model: $MODEL_ID"
echo "Using $NUM_GPUS GPUs for parallel execution."
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

        # --- MODIFIED PARALLEL EXECUTION LOGIC ---
        
        # 1. Count total images to divide the workload
        IMAGE_DIR_TO_SCAN="$IMAGE_PATH/samples"
        if [ ! -d "$IMAGE_DIR_TO_SCAN" ]; then
            IMAGE_DIR_TO_SCAN="$IMAGE_PATH"
        fi
        
        TOTAL_IMAGES=$(find "$IMAGE_DIR_TO_SCAN" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.webp" \) | wc -l)
        if [ "$TOTAL_IMAGES" -eq 0 ]; then
            echo "Warning: No images found in $IMAGE_DIR_TO_SCAN. Skipping category '$CATEGORY'."
            continue
        fi
        echo "Found $TOTAL_IMAGES images. Distributing workload across $NUM_GPUS GPUs."
        
        IMAGES_PER_GPU=$(( (TOTAL_IMAGES + NUM_GPUS - 1) / NUM_GPUS )) # Ceiling division
        PARTIAL_CSVS=()
        PARTIAL_LOGS=()

        # Handle special category argument for the Python script
        PYTHON_CATEGORY_ARG=$CATEGORY
        if [[ $CATEGORY == "complex action" || $CATEGORY == "complex spatial" ]]; then
            PYTHON_CATEGORY_ARG="complex"
        fi

        # 2. Launch background jobs for each GPU
        for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
            START_INDEX=$(( GPU_ID * IMAGES_PER_GPU ))
            END_INDEX=$(( START_INDEX + IMAGES_PER_GPU ))

            if [ "$START_INDEX" -ge "$TOTAL_IMAGES" ]; then
                continue # Skip creating jobs if there's no work for this GPU
            fi

            PARTIAL_OUTPUT_CSV="$OUTPUT_DIR/${CATEGORY_FILENAME_FRIENDLY}_${VARIANT_FILENAME_FRIENDLY}_gpu${GPU_ID}_results.csv"
            PARTIAL_LOG_FILE="$OUTPUT_DIR/${CATEGORY_FILENAME_FRIENDLY}_${VARIANT_FILENAME_FRIENDLY}_gpu${GPU_ID}_run.log"
            PARTIAL_CSVS+=("$PARTIAL_OUTPUT_CSV")
            PARTIAL_LOGS+=("$PARTIAL_LOG_FILE")

            echo "  -> Launching job on GPU $GPU_ID for images [$START_INDEX-$END_INDEX)"
            
            CUDA_VISIBLE_DEVICES=$GPU_ID python "$PYTHON_SCRIPT" \
                --image_path "$IMAGE_PATH" \
                --category "$PYTHON_CATEGORY_ARG" \
                --output_csv "$PARTIAL_OUTPUT_CSV" \
                --log_file "$PARTIAL_LOG_FILE" \
                --model_id "$MODEL_ID" \
                --batch_size "$BATCH_SIZE" \
                --start "$START_INDEX" \
                --end "$END_INDEX" &
        done

        # 3. Wait for all background jobs for this category to finish
        echo "Waiting for all jobs in '$CATEGORY' ($VARIANT) to complete..."
        wait
        echo "All jobs for '$CATEGORY' ($VARIANT) finished."

        # 4. Aggregate results from all partial files
        FINAL_OUTPUT_CSV="$OUTPUT_DIR/${CATEGORY_FILENAME_FRIENDLY}_${VARIANT_FILENAME_FRIENDLY}_results.csv"
        FINAL_LOG_FILE="$OUTPUT_DIR/${CATEGORY_FILENAME_FRIENDLY}_${VARIANT_FILENAME_FRIENDLY}_run.log"

        echo "Aggregating results into $FINAL_OUTPUT_CSV"
        # Smartly combine CSVs: print header from the first file, then all data rows from all files
        awk 'FNR==1 && NR!=1 {next} {print}' "${PARTIAL_CSVS[@]}" > "$FINAL_OUTPUT_CSV"
        # Combine logs for easier debugging
        cat "${PARTIAL_LOGS[@]}" > "$FINAL_LOG_FILE"

        # 5. Clean up partial files
        echo "Cleaning up partial files..."
        rm "${PARTIAL_CSVS[@]}"
        rm "${PARTIAL_LOGS[@]}"

        echo "Evaluation complete. Results saved in $FINAL_OUTPUT_CSV"
        echo "Log saved in $FINAL_LOG_FILE"

        # Calculate the mean score using the robust Python script
        if [ -f "$FINAL_OUTPUT_CSV" ]; then
            # Call the python script to calculate the mean from the 'score' column
            MEAN_SCORE=$(python calculate_mean.py "$FINAL_OUTPUT_CSV")
            echo "Calculated Mean Score for '$CATEGORY' ($VARIANT): $MEAN_SCORE"
            echo "$VARIANT,$CATEGORY,$MEAN_SCORE" >> "$SUMMARY_CSV"
        else
            echo "Warning: Output file $FINAL_OUTPUT_CSV not found. Cannot calculate mean score."
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