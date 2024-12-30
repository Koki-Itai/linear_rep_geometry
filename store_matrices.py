"""
概念ペアの文章を受け取り，Unembedding行列と標準化されたUnembedding行列に対して
"""
import torch
import numpy as np
import transformers
from tqdm import tqdm
import linear_rep_geometry as lrg
import os
import glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Store matrices for linear representation geometry analysis"
    )

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the pretrained model"
    )

    parser.add_argument(
        "--num_sample",
        type=int,
        default=1000,
        help="Number of random samples from dataset",
    )

    parser.add_argument(
        "--counterfactual_pair_txt_dir",
        type=str,
        required=True,
        help="Directory containing the dataset text files",
    )

    parser.add_argument(
        "--matrices_save_dir",
        type=str,
        required=True,
        help="Directory to save output matrices",
    )
    parser.add_argument("--prompt_type", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda:0")

    os.makedirs(args.matrices_save_dir, exist_ok=True)

    # init and load model
    print(f"Loading model: {args.model_path}")
    lrg.load_model(args.model_path)

    # load unembedding vectors
    print("load unembdding vectors ...")
    with torch.inference_mode():
        model = lrg.model
        gamma = model.lm_head.weight.detach() # Unembedding行列：[語彙サイズ × 隠れ層の次元数]の形状
        W, d = gamma.shape # W: 語彙サイズ，d: 隠れ層の次元数
        # 行列の中心化(行列の要素を原点に整列)
        gamma_bar = torch.mean(gamma, dim=0)
        centered_gamma = gamma - gamma_bar

    # 共分散行列の計算: Unembeddin行列の分布構造のキャプチャ
    print("compute Cov(gamma) and transform gamma to g ...")
    Cov_gamma = centered_gamma.T @ centered_gamma / W

    # 共分散行列の固有分解
    eigenvalues, eigenvectors = torch.linalg.eigh(Cov_gamma)

    # 逆平方根行列と平方根行列の計算
    inv_sqrt_Cov_gamma = (
        eigenvectors @ torch.diag(1 / torch.sqrt(eigenvalues)) @ eigenvectors.T
    )
    sqrt_Cov_gamma = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T

    # g: 標準化されたUnembedding行列
    g = gamma @ inv_sqrt_Cov_gamma

    # compute concept directions
    print("compute concept directions ...")
    filenames = sorted(glob.glob(os.path.join(args.counterfactual_pair_txt_dir, "*.txt")))

    concept_names = []
    for name in filenames:
        base = os.path.basename(name)
        content = os.path.splitext(base)[0][1:-1]
        parts = content.split(" - ")
        if len(parts) == 2:
            concept_names.append(r"${} \Rightarrow {}$".format(parts[0], parts[1]))
        else:
            concept_names.append(content)

    # 概念方向ベクトルの初期化
    concept_gamma = torch.zeros(len(filenames), d) # Unembedding空間での概念方向のベクトルを格納するテンソル
    concept_g = torch.zeros(len(filenames), d) # 標準化空間での概念方向

    count = 0

    # counter_factual pairの処理
    for filename in filenames:
        base_sequences, target_sequences, base_encodings, target_encodings = (
            lrg.get_counterfactual_pairs(
                filename, prompt_type=args.prompt_type, num_sample=int(args.num_sample)
            )
        )

        # 概念方向の計算
        mean_diff_gamma, diff_gamma = lrg.concept_direction(
            base_sequences, target_sequences
        )
        concept_gamma[count] = mean_diff_gamma

        mean_diff_g, diff_g = lrg.concept_direction(base_sequences, target_sequences)
        concept_g[count] = mean_diff_g

        count += 1

    # Save
    print("save everything ...")
    torch.save(gamma, f"{args.matrices_save_dir}/gamma.pt")
    print("Saved gamma.")
    torch.save(g, f"{args.matrices_save_dir}/g.pt")
    print("Saved g.")
    torch.save(sqrt_Cov_gamma, f"{args.matrices_save_dir}/sqrt_Cov_gamma.pt")
    print("Saved sqrt_Cov_gamma.")
    torch.save(concept_gamma, f"{args.matrices_save_dir}/concept_gamma.pt")
    print("Saved concept_gamma.")
    torch.save(concept_g, f"{args.matrices_save_dir}/concept_g.pt")
    print("Saved concept_g.")

    with open(f"{args.matrices_save_dir}/concept_names.txt", "w") as f:
        for item in concept_names:
            f.write(f"{item}\n")
    print("Saved concept_names.txt")

    with open(f"{args.matrices_save_dir}/filenames.txt", "w") as f:
        for item in filenames:
            f.write(f"{item}\n")
    print("Saved filenames.txt")


if __name__ == "__main__":
    main()
