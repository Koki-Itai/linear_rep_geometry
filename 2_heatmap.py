import warnings
warnings.filterwarnings('ignore')

import torch
import linear_rep_geometry as lrg
from torch.nn.functional import cosine_similarity
import argparse

device = torch.device("cuda:0")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrices_path", type=str, help="Path to the computed matrices"
    )
    parser.add_argument("--analyzed_figure_path", type=str)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    concept_gamma = torch.load(f'{args.matrices_path}/concept_gamma.pt').to(device)
    concept_g = torch.load(f'{args.matrices_path}/concept_g.pt').to(device)
    sqrt_Cov_gamma = torch.load(f"{args.matrices_path}/sqrt_Cov_gamma.pt").to(device)
    g = torch.load(f'{args.matrices_path}/g.pt').to(device)
    W, d = g.shape

    concept_names = []
    with open(f'{args.matrices_path}/concept_names.txt', 'r') as f:
        for index, line in enumerate(f.readlines()):
            concept_names.append(f"{line.strip()} ({index +1})")

    # compute the inner product between concept directions
    gamma_cosines = concept_gamma @ concept_gamma.T
    g_cosines = concept_g @ concept_g.T

    torch.manual_seed(100)
    another_g = concept_gamma @ torch.abs(torch.randn(d,d)).to(device)
    another_g_cosines = cosine_similarity(another_g.unsqueeze(1), another_g.unsqueeze(0), dim=-1)

    lrg.draw_heatmaps([torch.abs(g_cosines).cpu().numpy(),
                        torch.abs(gamma_cosines).cpu().numpy(),
                        torch.abs(another_g_cosines).cpu().numpy()],
                        concept_labels = concept_names,
                        cmap = "coolwarm",
                        save_dir=args.analyzed_figure_path)
    print(f"Saved at {args.analyzed_figure_path}.")