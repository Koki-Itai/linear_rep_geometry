import logging
import warnings
import torch
import linear_rep_geometry as lrg
import random
from datetime import datetime
import os
import argparse
import json
from tqdm import tqdm
from torch import nn
from transformers import PreTrainedTokenizer
from typing import List, Dict, Any
import transformers
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrices_path", type=str, help="Path to the computed matrices"
    )
    parser.add_argument("--analyzed_figure_path", type=str)
    parser.add_argument("--num_sample", type=str)
    parser.add_argument("--prompt_type", type=str)
    parser.add_argument("--random_txt_path", type=str, help="Path to random text file")
    parser.add_argument(
        "--generation_random_output_path", type=str, help="Path to save generation results"
    )
    parser.add_argument(
        "--generation_counterfactual_output_path", type=str, help="Path to save generation results"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument("--model_path", type=str)
    
    return parser.parse_args()


def get_sequence_pairs(filepath: str, num_samples: int = 1000) -> List[List[str]]:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) > num_samples:
        lines = random.sample(lines, num_samples)
    pairs = [line.split("\t") for line in lines]

    return pairs


def generate_text(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    prompt_type: str = "bare",
    max_new_tokens: int = 100,
    batch_size: int = 16,
    max_save_step: int = 1000
) -> List[str]:
    prompt_templates = {
        "reflection": "Consider the deeper meaning behind this: {}\nThis text reflects values related to:",
        "analysis": "Analyzing the underlying principles in: {}\nThe core values expressed here are:",
        "implicit": "Looking at the implicit meaning of: {}\nThe fundamental values shown are:",
        "explicit": "What values are represented in this text: {}\nThe text demonstrates values of:",
        "bare": "{}",
        "theme": "What is the main theme in this text: {}\nKey themes include:",
        "topic": "Identify the primary topics discussed in: {}\nMain topics covered are:",
    }
    
    if prompt_type not in prompt_templates:
        raise ValueError(
            f"Invalid prompt_type: {prompt_type}. Available types: {list(prompt_templates.keys())}"
        )
    
    template = prompt_templates[prompt_type]
    formatted_prompts = [template.format(prompt) for prompt in prompts]
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = 'left'
    generated_texts = []
    
    # max_save_stepをバッチサイズで考慮
    effective_max_save_step = max_save_step // batch_size * batch_size

    for i in tqdm(range(0, len(formatted_prompts), batch_size), desc="Generating texts", total=len(formatted_prompts)//batch_size + bool(len(formatted_prompts)%batch_size)):
        if i > effective_max_save_step:
            break
        batch_prompts = formatted_prompts[i:i + batch_size]
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7,
            )

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_inputs = tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=True)
        
        batch_generated = [
            output[len(input_text):].strip()
            for output, input_text in zip(decoded_outputs, decoded_inputs)
        ]
        generated_texts.extend(batch_generated)
    
    return generated_texts

def save_generation_results(
    pairs: List[List[str]],
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    output_path: str,
    concept_name: str,
    prompt_type: str,
    max_new_tokens: int = 100,
    batch_size: int = 8,
    max_save_step: int = 1000,
) -> None:
    concept_texts = [pair[0] for pair in pairs]
    non_concept_texts = [pair[1] for pair in pairs]
    
    generated_concept_texts = generate_text(
        model, tokenizer, concept_texts, prompt_type, max_new_tokens, batch_size, max_save_step
    )
    generated_non_concept_texts = generate_text(
        model, tokenizer, non_concept_texts, prompt_type, max_new_tokens, batch_size, max_save_step
    )
    
    results = [
        {
            "concept_text": c_text,
            "non_concept_text": nc_text,
            "generated_text_concept": gc_text,
            "generated_text_non_concept": gnc_text,
        }
        for c_text, nc_text, gc_text, gnc_text in zip(
            concept_texts, non_concept_texts, 
            generated_concept_texts, generated_non_concept_texts
        )
    ]

    output_path = output_path.split(".json")[0] + f"_{concept_name}.json"
    print(f"{output_path=}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    args = parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    warnings.filterwarnings("ignore")
    device = torch.device("cuda:0")

    # Load model and tokenizer from linear_rep_geometry
    lrg.load_model(model_path=args.model_path)
    model = lrg.model
    tokenizer = lrg.tokenizer

    matrices_path = args.matrices_path
    g = torch.load(f"{matrices_path}/g.pt").to(device)
    concept_g = torch.load(f"{matrices_path}/concept_g.pt").to(device)
    W, d = g.shape

    filenames = []
    with open(f"{matrices_path}/filenames.txt", "r") as f:
        for line in f.readlines():
            filenames.append(line.strip())

    concept_names = []
    with open(f"{matrices_path}/concept_names.txt", "r") as f:
        for line in f.readlines():
            concept_names.append(line.strip())

    # Get random pairs from file
    print("Processing random pairs from file...")
    random_pairs = get_sequence_pairs(args.random_txt_path, int(args.num_sample))

    if args.generation_random_output_path:
        print("Generating and saving model outputs (Random pairs text) ...")
        generation_results = save_generation_results(
            pairs=random_pairs,
            model=model,
            tokenizer=tokenizer,
            concept_name="",
            prompt_type=args.prompt_type,
            output_path=args.generation_random_output_path,
            max_new_tokens=int(args.max_new_tokens),
            max_save_step=10
        )

    # Get embeddings for random pairs
    random_concept_sequences = [pair[0] for pair in random_pairs]
    random_non_concept_sequences = [pair[1] for pair in random_pairs]

    # compute the projections on concept directions
    inner_product_with_counterfactual_pairs_g_LOO = []

    num_sample = int(args.num_sample)
    count = 0
    for filename in filenames:
        print(f"Processing {filename} ...")
        concept_sequences, non_concept_sequences, concept_encodings, non_concept_encodings = (
            lrg.get_counterfactual_pairs(
                filename, prompt_type=args.prompt_type, num_sample=num_sample
            )
        )
        inner_product_LOO, diff_data = lrg.inner_product_loo(
            concept_sequences, non_concept_sequences, g
        )
        inner_product_with_counterfactual_pairs_g_LOO.append(inner_product_LOO)
        count += 1

        ## Generate next tokens 
        counterfactual_pairs = get_sequence_pairs(filename, num_samples=int(args.num_sample))
        pattern = r'\[(.*?)\('
        match1 = re.search(pattern, filename.split("/")[-1])
        concept_name = match1.group(1)
        if args.generation_random_output_path:
            print("Generating and saving model outputs (Counterfactual pairs text) ...")
            generation_results = save_generation_results(
                pairs=counterfactual_pairs,
                model=model,
                tokenizer=tokenizer,
                prompt_type=args.prompt_type,
                output_path=args.generation_counterfactual_output_path,
                max_new_tokens=int(args.max_new_tokens),
                concept_name=concept_name,
                max_save_step=10,
            )



    # Get embeddings and calculate differences for random pairs
    print("Computing embeddings for random pairs...")
    random_concept_embeddings = lrg.get_embeddings(random_concept_sequences, batch_size=2)
    random_non_concept_embeddings = lrg.get_embeddings(random_non_concept_sequences, batch_size=2)
    random_pairs_g = random_non_concept_embeddings - random_concept_embeddings

    lrg.show_histogram_LOO(
        inner_product_with_counterfactual_pairs_g_LOO,
        random_pairs_g,
        concept_g,
        concept_names,
        fig_name="g",
        save_dir=args.analyzed_figure_path,
    )
