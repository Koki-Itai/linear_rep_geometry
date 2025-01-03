import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import os
import numpy as np
from typing import List, Dict, Union, Optional
from torch import Tensor, device as torch_device
from tqdm import tqdm
import linear_rep_geometry as lrg
import os
import numpy as np
import matplotlib.pyplot as plt
import transformers
import matplotlib.gridspec as gridspec
import re

def run_intervention(
    model: torch.nn.Module,
    tokenizer: Union[torch.nn.Module, object],
    texts: List[str],
    concept_g: Tensor,
    concept2idx: Dict[str, int],
    intervention_value_name: str = "Achievement",
    min_alpha: float = 0.0,
    max_alpha: float = 1.0,
    step_alpha: float = 0.1,
    n_generate_tokens: int = 32,
    temperature: float = 0.8,
    top_p: float = 0.95,
    device: torch_device = "cuda",
    verbose: bool = False,
    save_output: bool = True
) -> np.ndarray:
    intervention_value_idx = concept2idx[intervention_value_name]
    concept_vec = concept_g[intervention_value_idx]
    alphas = np.arange(min_alpha, max_alpha + step_alpha, step_alpha)
    
    if save_output:
        os.makedirs("intervention", exist_ok=True)
        output_path = os.path.join("intervention", f"{intervention_value_name}.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# {intervention_value_name}\n\n")

    total_iterations = len(alphas) * len(texts)
    pbar = tqdm(total=total_iterations, desc="Generating interventions")

    for alpha in alphas:
        results = []
        for text in texts:
            prompt = f"Please write a natural continuation following this text:\n{text}"
            encoded = tokenizer(
                prompt,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
                max_length=128,
            ).to(device)

            input_ids = encoded.input_ids
            attention_mask = encoded.attention_mask
            generated_ids = input_ids

            for _ in range(n_generate_tokens):
                with torch.no_grad():
                    outputs = model(
                        generated_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                    
                    last_hidden_state = outputs.hidden_states[-1]
                    last_token_state = last_hidden_state[:, -1, :]
                    intervened_state = last_token_state + alpha * concept_vec
                    logits = model.lm_head(intervened_state)
                    
                    scaled_logits = logits[0] / temperature
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    sorted_logits[sorted_indices_to_remove] = float('-inf')
                    
                    probs_to_sample = torch.zeros_like(scaled_logits)
                    probs_to_sample[sorted_indices] = sorted_logits
                    final_probs = F.softmax(probs_to_sample, dim=-1)
                    
                    next_token_id = torch.multinomial(final_probs, num_samples=1).unsqueeze(0)
                    
                    generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((attention_mask.shape[0], 1), device=device)],
                        dim=-1,
                    )

            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            original_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            continuation = generated_text[len(original_prompt):].strip()
            full_text = text + " " + continuation
            
            results.append({
                "original": text,
                "interventioned": full_text
            })

            if verbose:
                print(f"Alpha: {alpha:.1f}, Text: {text}")
                print(f"Generated: {full_text}")
            
            pbar.update(1)

        if save_output:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(f"\n## α = {alpha:.1f}\n\n")
                for i, result in enumerate(results, 1):
                    f.write(f"### Text {i}\n")
                    f.write(f"**Original**: {result['original']}\n\n")
                    f.write(f"**Interventioned**: {result['interventioned']}\n\n")
                f.write("---\n")
    
    pbar.close()
    print("Complete All Generation.")

if __name__ == "__main__":
    device_id = 2
    device = torch.device(f"cuda:{device_id}")

    g = torch.load('tmp_matrices/g.pt').to(device)
    concept_g = torch.load('tmp_matrices/concept_g.pt').to(device)
    sqrt_Cov_gamma = torch.load("tmp_matrices/sqrt_Cov_gamma.pt").to(device)
    W, d = g.shape

    concept_names = []
    with open('tmp_matrices/concept_names.txt', 'r') as f:
        for line in f.readlines():
            concept_names.append(line.strip())

    model_path = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": device},
    )
    model = model.to(device)

    neutral_texts = [
        "The person completed the task.",
        "They made a decision about the matter.",
        "The group discussed the situation.",
        "telling my friend that it will take 7 years for her to graduate with her double major",
        "My colleague is quitting her job."
    ]

    # ターゲットの価値観名を取得，idxと紐付け
    filenames = []
    with open(f"tmp_matrices/filenames.txt", "r") as f:
        for line in f.readlines():
            filenames.append(line.strip())

    concept2idx = {}
    idx2concept = {}
    pattern = r'\[(.*?)\('
    for idx, filename in enumerate(filenames):
        match1 = re.search(pattern, filenames[idx].split("/")[-1])
        concept_name = match1.group(1)
        concept2idx[concept_name] = idx
        idx2concept[idx] = concept_name

    # Intervention
    for value_name in concept2idx.keys():
        print(f"Value Name: {value_name}")
        run_intervention(
            model=model,
            tokenizer=tokenizer,
            texts=neutral_texts,
            concept_g=concept_g,
            concept2idx=concept2idx,
            intervention_value_name=value_name,
            min_alpha=0.0,
            max_alpha=1.0,
            step_alpha=0.1,
            n_generate_tokens=64,
            temperature=0.9,
            top_p=0.95,
            device=device,
            verbose=False,
            save_output=True
        )