# Experiments config
MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct" # [meta-llama/Llama-3.1-8B-Instruct, meta-llama/Llama-2-7b-hf, meta-llama/Llama-3.2-3B-Instruct]
DATASET_TYPE="valuenet" # [valuenet]
CONCEPT_DIRECTION="pos2neg" # [pos2neg, pos2neutral]
NORM_TYPE="norm_sentence_structure" # [base, norm_sentence_structure, norm_topic]
PROMPT_TYPE="bare"  # [bare, reflection, analysis, implicit, explicit]

# Basic config
MODEL_NAME=$(echo $MODEL_PATH | cut -d'/' -f2 | tr '[:upper:]' '[:lower:]')
NUM_SAMPLE=1000
SAVE_DIR="tmp_matrices"
COUTER_FACTUAL_TXT_DIR="/home/itai/research/linear_rep_geometry/data/ValueNet/schwartz/${CONCEPT_DIRECTION}/${NORM_TYPE}"
ANALYZED_FIGURE_DIR="/home/itai/research/linear_rep_geometry/figures/${MODEL_NAME}/${DATASET_TYPE}/${CONCEPT_DIRECTION}/${NORM_TYPE}/${PROMPT_TYPE}"
GENERATION_OUTPUT_PATH="/home/itai/research/linear_rep_geometry/generated/${MODEL_NAME}/${DATASET_TYPE}/${CONCEPT_DIRECTION}/${NORM_TYPE}/${PROMPT_TYPE}"
RANDOM_TXT_PATH="/home/itai/research/linear_rep_geometry/data/ValueNet/schwartz/random_pairs/${NORM_TYPE}"

echo "=== Create Matrices ... ==="
python store_matrices.py \
    --model_path $MODEL_PATH \
    --counterfactual_pair_COUTER_FACTUAL_TXT_DIR $COUTER_FACTUAL_TXT_DIR \
    --matrices_save_dir $SAVE_DIR \
    --num_sample $NUM_SAMPLE \
    --prompt_type $PROMPT_TYPE

echo "=== Analyze Subspace ... ==="
python 1_subspace.py \
    --matrices_path $SAVE_DIR \
    --num_sample $NUM_SAMPLE \
    --analyzed_figure_path $ANALYZED_FIGURE_DIR \
    --prompt_type $PROMPT_TYPE \
    --random_txt_path $RANDOM_TXT_PATH \
    --generation_output_path $GENERATION_OUTPUT_PATH