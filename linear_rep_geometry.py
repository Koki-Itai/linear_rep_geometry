import transformers
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import json
from tqdm import tqdm
import random
import os
from torch.nn import DataParallel

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda:0")
torch.cuda.set_device(device)

sns.set_theme(
    context="paper",
    style="white",  # 'whitegrid', 'dark', 'darkgrid', ...
    palette="colorblind",
    font="sans-serif",  # 'serif'
    font_scale=1.75,  # 1.75, 2, ...
)


def initialize_model(model_path=None):
    if model_path is None:
        # default
        model_path = "meta-llama/Llama-3.2-3B-Instruct"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": device},
    )
    model = model.to(device)

    return tokenizer, model


def load_model(model_path=None):
    global tokenizer, model
    tokenizer, model = initialize_model(model_path)


tokenizer = None
model = None

# MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"
# tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
# model = transformers.AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     device_map={"": device},
# )
# model = model.to(device)


## get indices of counterfactual pairs
def get_counterfactual_pairs(filename, prompt_type, num_sample=1000):
    prompt_templates = {
        "reflection": "Consider the deeper meaning behind this: {}\nThis text reflects values related to:",
        "analysis": "Analyzing the underlying principles in: {}\nThe core values expressed here are:",
        "implicit": "Looking at the implicit meaning of: {}\nThe fundamental values shown are:",
        "explicit": "What values are represented in this text: {}\nThe text demonstrates values of:",
        "bare": "{}",
    }
    if prompt_type not in prompt_templates:
        raise ValueError(
            f"Invalid prompt_type: {prompt_type}. Available types: {list(prompt_templates.keys())}"
        )
    template = prompt_templates[prompt_type]

    with open(filename, "r") as f:
        lines = f.readlines()

    if len(lines) > num_sample:
        lines = random.sample(lines, num_sample)

    text_pairs = []
    for line in lines:
        if line.strip():
            base, target = line.strip().split("\t")
            prefixed_base = template.format(base) if template else base
            prefixed_target = template.format(target) if template else target
            text_pairs.append((prefixed_base, prefixed_target))

    base_sequences = []
    target_sequences = []
    base_encodings = []
    target_encodings = []

    for base, target in text_pairs:
        first_encoding = tokenizer.encode(base, add_special_tokens=True)
        second_encoding = tokenizer.encode(target, add_special_tokens=True)

        if len(first_encoding) >= 1 and len(second_encoding) >= 1:
            base_sequences.append(base)
            target_sequences.append(target)
            base_encodings.append(first_encoding)
            target_encodings.append(second_encoding)

    return base_sequences, target_sequences, base_encodings, target_encodings


def concept_direction(base_sequences, target_sequences):
    """単語のみ（元論文の実装）"""
    #     base_data = data[base_ind,]
    #     target_data = data[target_ind,]

    #     diff_data = target_data - base_data
    #     mean_diff_data = torch.mean(diff_data, dim=0)
    #     mean_diff_data = mean_diff_data / torch.norm(mean_diff_data)

    #     return mean_diff_data, diff_data

    """トークン列に対応"""
    print("Computing base embeddings...")
    base_embeddings = get_embeddings(base_sequences)

    print("Computing target embeddings...")
    target_embeddings = get_embeddings(target_sequences)

    print("Calculating concept direction...")
    diff_vectors = target_embeddings - base_embeddings
    mean_diff = torch.mean(diff_vectors, dim=0)
    concept_vector = mean_diff / torch.norm(mean_diff)

    return concept_vector, diff_vectors


model = DataParallel(model)


def get_embeddings(text_batch, batch_size=4):
    """データパラレルによりGPU効率をあげる実装"""
    global model, tokenizer
    if model is None or tokenizer is None:
        load_model()

    device = next(model.parameters()).device
    tokenizer.pad_token = tokenizer.eos_token
    all_embeddings = []

    total_batches = (len(text_batch) + batch_size - 1) // batch_size
    pbar = tqdm(
        range(0, len(text_batch), batch_size),
        total=total_batches,
        desc="Processing embeddings",
        unit="batch",
    )
    try:
        for i in pbar:
            batch_texts = text_batch[i : i + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
                max_length=128,
            )

            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)

            torch.cuda.empty_cache()

            with torch.no_grad():
                outputs = model(
                    input_ids, attention_mask=attention_mask, output_hidden_states=True
                )
                last_hidden_state = outputs.hidden_states[-1]
                masked_hidden_state = last_hidden_state * attention_mask.unsqueeze(-1)
                sum_hidden_state = masked_hidden_state.sum(dim=1)
                count_tokens = attention_mask.sum(dim=1).unsqueeze(-1)
                batch_embeddings = sum_hidden_state / count_tokens

                all_embeddings.append(batch_embeddings.cpu())

                del outputs, last_hidden_state, masked_hidden_state, sum_hidden_state
                torch.cuda.empty_cache()

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(
                f"WARNING: Out of memory error occurred. Try reducing batch_size (current: {batch_size})"
            )
            torch.cuda.empty_cache()
        raise e

    combined_embeddings = torch.cat(all_embeddings, dim=0).to(device)
    return combined_embeddings


# def get_embeddings(text_batch, batch_size=4):
#     """
#     Get embeddings for text sequences in batches with progress visualization.

#     Args:
#         text_batch: List of input texts
#         batch_size: Number of texts to process at once

#     Returns:
#         torch.Tensor: Embeddings for all input texts
#     """
#     global model, tokenizer
#     if model is None or tokenizer is None:
#         load_model()

#     device = next(model.parameters()).device
#     tokenizer.pad_token = tokenizer.eos_token
#     all_embeddings = []

#     # Calculate total number of batches
#     total_batches = (len(text_batch) + batch_size - 1) // batch_size

#     # Create progress bar
#     pbar = tqdm(
#         range(0, len(text_batch), batch_size),
#         total=total_batches,
#         desc="Processing embeddings",
#         unit="batch",
#     )
#     try:
#         for i in pbar:
#             batch_texts = text_batch[i : i + batch_size]

#             # Update progress bar with batch size info
#             pbar.set_postfix(
#                 {
#                     "batch_size": len(batch_texts),
#                     "processed": min(i + batch_size, len(text_batch)),
#                 }
#             )

#             encoded = tokenizer(
#                 batch_texts,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 return_attention_mask=True,
#                 max_length=128,
#             )

#             input_ids = encoded.input_ids.to(device)
#             attention_mask = encoded.attention_mask.to(device)

#             torch.cuda.empty_cache()

#             with torch.no_grad():
#                 outputs = model(
#                     input_ids, attention_mask=attention_mask, output_hidden_states=True
#                 )

#                 last_hidden_state = outputs.hidden_states[-1]
#                 masked_hidden_state = last_hidden_state * attention_mask.unsqueeze(-1)
#                 sum_hidden_state = masked_hidden_state.sum(dim=1)
#                 count_tokens = attention_mask.sum(dim=1).unsqueeze(-1)
#                 batch_embeddings = sum_hidden_state / count_tokens

#                 all_embeddings.append(batch_embeddings.cpu())

#                 del outputs, last_hidden_state, masked_hidden_state, sum_hidden_state
#                 torch.cuda.empty_cache()

#     except RuntimeError as e:
#         if "out of memory" in str(e):
#             print(f"WARNING: Out of memory error occurred. Try reducing batch_size (current: {batch_size})")
#             torch.cuda.empty_cache()
#         raise e

#     combined_embeddings = torch.cat(all_embeddings, dim=0).to(device)
#     return combined_embeddings


####### Experiment 1: subspace #######
def inner_product_loo(base_sequences, target_sequences, data):
    """単語に対応（元論文の実装）"""
    #     base_data = data[base_ind,]
    #     target_data = data[target_ind,]

    #     diff_data = target_data - base_data
    #     products = []
    #     for i in range(diff_data.shape[0]):
    #         mask = torch.ones(diff_data.shape[0], dtype=bool)
    #         mask[i] = False
    #         loo_diff = diff_data[mask]
    #         mean_diff_data = torch.mean(loo_diff, dim=0)
    #         loo_mean = mean_diff_data / torch.norm(mean_diff_data)
    #         products.append(loo_mean @ diff_data[i])
    #     return torch.stack(products), diff_data

    """トークン列に対応"""
    base_data = get_embeddings(base_sequences)
    target_data = get_embeddings(target_sequences)

    diff_data = target_data - base_data

    # Leave-One-Out法による内積計算
    products = []
    for i in range(diff_data.shape[0]):
        mask = torch.ones(diff_data.shape[0], dtype=torch.bool, device=diff_data.device)
        mask[i] = False
        loo_diff = diff_data[mask]
        mean_diff_data = torch.mean(loo_diff, dim=0)
        loo_mean = mean_diff_data / torch.norm(mean_diff_data)
        products.append(loo_mean @ diff_data[i])

    return torch.stack(products), diff_data


def show_histogram_LOO(
    inner_product_with_counterfactual_pairs_LOO,
    random_pairs,
    concept,
    concept_names,
    save_dir,
    fig_name="gamma",
    cols=4,
    title_fontsize=12,
):
    n_plots = concept.shape[0]
    cols = min(cols, n_plots)
    rows = (n_plots + cols - 1) // cols

    fig_height = 6 * rows
    fig_width = 5 * cols
    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = fig.add_gridspec(rows, cols, hspace=0.6, wspace=0.4)
    axs = []
    for i in range(rows):
        for j in range(cols):
            axs.append(fig.add_subplot(gs[i, j]))

    for i in range(len(axs)):
        if i < n_plots:
            target = inner_product_with_counterfactual_pairs_LOO[i]
            baseline = random_pairs @ concept[i]

            axs[i].hist(
                baseline.cpu().numpy(),
                bins=40,
                alpha=0.6,
                color="blue",
                label="random pairs",
                density=True,
            )
            axs[i].hist(
                target.cpu().numpy(),
                bins=40,
                alpha=0.7,
                color="red",
                label="counterfactual pairs",
                density=True,
            )
            axs[i].set_yticks([])

            title = concept_names[i]
            axs[i].set_title(title, fontsize=title_fontsize, pad=15)

            axs[i].tick_params(axis="x", labelrotation=45, labelsize=10)
            axs[i].grid(True, alpha=0.3)

            axs[i].spines["top"].set_visible(False)
            axs[i].spines["right"].set_visible(False)
            axs[i].spines["left"].set_visible(False)
        else:
            axs[i].axis("off")

    handles, labels = axs[0].get_legend_handles_labels()
    if n_plots < len(axs):
        legend_ax = axs[n_plots]
    else:
        legend_ax = axs[-1]
    legend_ax.legend(handles, labels, loc="center", fontsize=12, frameon=False)
    legend_ax.axis("off")

    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(
        os.path.join(save_dir, f"appendix_right-skewed_LOO_{fig_name}.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


####### Experiment 2: heatmap #######
def draw_heatmaps(data_matrices, concept_labels, cmap="PiYG"):
    fig = plt.figure(figsize=(14, 8.5))
    gs = gridspec.GridSpec(2, 3, wspace=0.2)

    vmin = min([data.min() for data in data_matrices])
    vmax = max([data.max() for data in data_matrices])

    ticks = list(range(2, 27, 3))
    labels = [str(i + 1) for i in ticks]

    ytick = list(range(27))
    ims = []

    ax_left = plt.subplot(gs[0:2, 0:2])
    im = ax_left.imshow(data_matrices[0], cmap=cmap)
    ims.append(im)
    ax_left.set_xticks(ticks)
    ax_left.set_xticklabels(labels)
    ax_left.set_yticks(ytick)
    ax_left.set_yticklabels(concept_labels)
    ax_left.set_title(r"$M = \mathrm{Cov}(\gamma)^{-1}$")

    ax_top_right = plt.subplot(gs[0, 2])
    im = ax_top_right.imshow(data_matrices[1], cmap=cmap)
    ims.append(im)
    ax_top_right.set_xticks([])
    ax_top_right.set_yticks([])
    ax_top_right.set_title(r"$M = I_d$")

    ax_bottom_right = plt.subplot(gs[1, 2])
    im = ax_bottom_right.imshow(data_matrices[2], cmap=cmap)
    ims.append(im)
    ax_bottom_right.set_xticks([])
    ax_bottom_right.set_yticks([])
    ax_bottom_right.set_title(r"Random $M$")

    divider = make_axes_locatable(ax_left)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(ims[-1], cax=cax, orientation="vertical")

    plt.tight_layout()
    plt.savefig(f"figures/three_heatmaps.png", bbox_inches="tight")
    plt.show()


####### Experiment 3: measurement #######
def get_lambda_pairs(filename, num_eg=20):
    lambdas_0 = []
    lambdas_1 = []

    count = 0
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # Data Sampling
        if len(lines) > 1000:
            lines = random.sample(lines, 1000)

        for line in tqdm(lines):
            data = json.loads(line)
            if count >= num_eg:
                break

            text_0 = [s.strip(" " + data["word0"]) for s in data["contexts0"]]
            lambdas_0.append(get_embeddings(text_0))

            text_1 = [s.strip(" " + data["word1"]) for s in data["contexts1"]]
            lambdas_1.append(get_embeddings(text_1))

            count += 1

    return torch.cat(lambdas_0), torch.cat(lambdas_1)


def hist_measurement(
    lambda_0,
    lambda_1,
    concept,
    concept_names,
    base="English",
    target="French",
    alpha=0.5,
):
    fig, axs = plt.subplots(7, 4, figsize=(16, 20))

    axs = axs.flatten()

    for i in range(concept.shape[0]):
        W0 = lambda_0 @ concept[i]
        W1 = lambda_1 @ concept[i]

        axs[i].hist(W0.cpu().numpy(), bins=25, alpha=alpha, label=base, density=True)
        axs[i].hist(W1.cpu().numpy(), bins=25, alpha=alpha, label=target, density=True)
        axs[i].set_yticks([])
        axs[i].set_title(f"{concept_names[i]}")

    handles, labels = axs[0].get_legend_handles_labels()
    axs[concept.shape[0]].legend(handles, labels, loc="center")
    axs[concept.shape[0]].axis("off")

    plt.tight_layout()
    plt.savefig(
        "figures/appendix_measurement_" + base + "-" + target + ".png",
        bbox_inches="tight",
    )
    plt.show()


####### Experiment 4: intervention #######
def get_logit(embedding, unembedding, base="king", W="queen", Z="King"):
    num = embedding.shape[0]
    logit = torch.zeros(num, 2)
    for i in range(num):
        index_base = tokenizer.encode(base)[1]
        index_W = tokenizer.encode(W)[1]
        index_Z = tokenizer.encode(Z)[1]
        value = unembedding @ embedding[i]
        logit[i, 0] = value[index_W] - value[index_base]
        logit[i, 1] = value[index_Z] - value[index_base]
    return logit


def show_arrows(
    logit_original,
    logit_intervened_l,
    concept_names,
    base="king",
    W="queen",
    Z="King",
    xlim=[-15, 5],
    ylim=[-15, 7],
    fig_name="gamma",
):
    fig, axs = plt.subplots(7, 4, figsize=(16, 20))

    axs = axs.flatten()

    for i in range(len(concept_names)):
        origin = logit_original.numpy()
        vectors_A = logit_intervened_l[i].numpy() - logit_original.numpy()

        axs[i].quiver(
            *origin.T,
            vectors_A[:, 0],
            vectors_A[:, 1],
            color="b",
            angles="xy",
            scale_units="xy",
            scale=1,
            label="intervened lambda",
            alpha=1,
        )

        axs[i].set_xlim(xlim)
        axs[i].set_ylim(ylim)
        axs[i].grid(True, linestyle="--", alpha=0.7)
        axs[i].set_title(f"{concept_names[i]}")

    handles, labels = axs[0].get_legend_handles_labels()
    axs[len(concept_names)].legend(handles, labels, loc="center")
    axs[len(concept_names)].set_yticklabels([])
    axs[len(concept_names)].set_xticklabels([])

    plt.xlabel(
        rf"$\log\frac{{\mathbb{{P}}({W}\mid\lambda)}}{{\mathbb{{P}}({base}\mid \lambda)}}$"
    )
    plt.ylabel(
        rf"$\log\frac{{\mathbb{{P}}({Z}\mid \lambda)}}{{\mathbb{{P}}({base}\mid \lambda)}}$"
    )

    plt.tight_layout()
    plt.savefig(
        "figures/appendix_intervention_"
        + fig_name
        + "_"
        + base
        + "_"
        + W
        + "_"
        + Z
        + ".png",
        bbox_inches="tight",
    )
    plt.show()


def show_intervention(
    embedding_batch,
    unembedding,
    concept,
    concept_names,
    base="king",
    W="queen",
    Z="King",
    alpha=0.5,
    xlim=[-15, 15],
    ylim=[-10, 10],
    fig_name="gamma",
):
    logit_original = get_logit(embedding_batch, unembedding, base=base, W=W, Z=Z)
    logit_intervened_embedding = []

    for i in range(len(concept_names)):
        intervened_embedding = embedding_batch + alpha * concept[i]
        logit_intervened_embedding.append(
            get_logit(intervened_embedding, unembedding, base=base, W=W, Z=Z)
        )
    show_arrows(
        logit_original,
        logit_intervened_embedding,
        concept_names,
        base=base,
        W=W,
        Z=Z,
        xlim=xlim,
        ylim=ylim,
        fig_name=fig_name,
    )


def show_rank(text_batch, l_batch, g, concept_g, which_ind, concept_number):
    alphas = torch.linspace(0, 0.4, 5)
    print("Prompt:", text_batch[which_ind])
    print("=" * 40)
    l_king = l_batch[which_ind]

    top_k = 5

    top_tokens = [[] for _ in alphas]
    for i in range(len(alphas)):
        new_lambda = l_king + alphas[i] * concept_g[concept_number]
        value = g @ new_lambda
        norm_values, norm_indices = torch.topk(value, k=top_k, largest=True)
        for j in range(top_k):
            top_tokens[i].append(tokenizer.decode(norm_indices[j]))

    print("  & ", " & ".join([str(alpha.item()) for alpha in alphas]))
    for j in range(top_k):
        print(j + 1, "&", " & ".join([token_row[j] for token_row in top_tokens]))


def sanity_check(
    g,
    concept_g,
    a_i,
    b_i,
    c_i,
    d_i,
    concept_names,
    alpha=0.3,
    s=0.8,
    name_1=[],
    ind_1=[],
    name_2=[],
    ind_2=[],
):
    a_g = g @ concept_g[a_i]
    b_g = g @ concept_g[b_i]
    c_g = g @ concept_g[c_i]
    d_g = g @ concept_g[d_i]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].scatter(a_g.cpu().numpy(), b_g.cpu().numpy(), alpha=alpha, s=s)
    for _, label in enumerate(name_1):
        axs[0].text(a_g[ind_1[_]], b_g[ind_1[_]], label, fontsize=12)
    axs[0].set_xlabel(r"$\bar{\lambda}_W^\top \gamma$")
    axs[0].set_ylabel(r"$\bar{\lambda}_Z^\top \gamma$")
    axs[0].set_title(
        f"W: {concept_names[a_i]}, Z: {concept_names[b_i]}", x=0.48, y=1.02
    )

    axs[1].scatter(c_g.cpu().numpy(), d_g.cpu().numpy(), alpha=alpha, s=s)
    for _, label in enumerate(name_2):
        axs[1].text(c_g[ind_2[_]], d_g[ind_2[_]], label, fontsize=12)
    axs[1].set_xlabel(r"$\bar{\lambda}_W^\top \gamma$")
    axs[1].set_ylabel(r"$\bar{\lambda}_Z^\top \gamma$")
    axs[1].set_title(
        f"W: {concept_names[c_i]}, Z: {concept_names[d_i]}", x=0.48, y=1.02
    )

    axs[0].axhline(0, color="gray", linestyle="--", alpha=0.6)
    axs[0].axvline(0, color="gray", linestyle="--", alpha=0.6)
    axs[1].axhline(0, color="gray", linestyle="--", alpha=0.6)
    axs[1].axvline(0, color="gray", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("figures/sanity_check.png", dpi=300, bbox_inches="tight")
    plt.show()
