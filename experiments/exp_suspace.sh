#!/bin/bash

# Setup
bash /home/itai/research/linear_rep_geometry/setup.sh

# Config
MODEL_PATHS=("meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-2-7b-hf")
DATASET_TYPES=("valuenet")
CONCEPT_DIRECTIONS=("pos2neg" "pos2neutral")
NORM_TYPES=("base" "norm_sentence_structure")
PROMPT_TYPES=("bare" "reflection" "analysis" "implicit" "explicit")

LOG_DIR="experiment_logs"
mkdir -p $LOG_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

run_experiment() {
    local model_path=$1
    local dataset_type=$2
    local concept_direction=$3
    local norm_type=$4
    local prompt_type=$5

    local log_file="${LOGDIR}/experiment${TIMESTAMP}_${modelpath##*/}${datasettype}${conceptdirection}${normtype}${prompt_type}.log"

    echo "* ===== Running experiment with configuration: ===== "
    echo "Model: $model_path"
    echo "Dataset: $dataset_type"
    echo "Concept Direction: $concept_direction"
    echo "Norm Type: $norm_type"
    echo "Prompt Type: $prompt_type"
    echo "Log file: $log_file"
    echo " ================================================= *"

    export MODEL_PATH=$model_path
    export DATASET_TYPE=$dataset_type
    export CONCEPT_DIRECTION=$concept_direction
    export NORM_TYPE=$norm_type
    export PROMPT_TYPE=$prompt_type

    ./1_subspace.sh 2>&1 | tee "$log_file"

    echo "Experiment completed. Log saved to: $log_file"
    echo "----------------------------------------"
}

total_experiments=$((${#MODEL_PATHS[@]} * ${#DATASET_TYPES[@]} * ${#CONCEPT_DIRECTIONS[@]} * ${#NORM_TYPES[@]} * ${#PROMPT_TYPES[@]}))
current_experiment=0

for model_path in "${MODEL_PATHS[@]}"; do
    for dataset_type in "${DATASET_TYPES[@]}"; do
        for concept_direction in "${CONCEPT_DIRECTIONS[@]}"; do
            for norm_type in "${NORM_TYPES[@]}"; do
                for prompt_type in "${PROMPT_TYPES[@]}"; do
                    ((current_experiment++))
                    echo "Starting experiment $current_experiment of $total_experiments"
                    run_experiment "$model_path" "$dataset_type" "$concept_direction" "$norm_type" "$prompt_type"
                done
            done
        done
    done
done

echo "All experiments completed!"
echo "Logs are available in: $LOG_DIR"
