import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import argparse
from univ_utils import load_model_and_sae, get_running_activation_stats, load_data
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n1", type=int, default=0)
    parser.add_argument("--n2", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_batches", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

args = parse_args()

assert args.n1 != args.n2

model_sae_pairs = [
    ("8-768", "443ngubm"),
    ("8-512", "fyqbawtf"),
    ("8-256", "7g6hq05j"),
    ("8-128", "ngd29532"),
]

os.makedirs("stats_data", exist_ok=True)


model1_name, sae1_name = model_sae_pairs[args.n1]
model2_name, sae2_name = model_sae_pairs[args.n2]

model1, sae1 = load_model_and_sae(model1_name, sae1_name, None, device)
model2, sae2 = load_model_and_sae(model2_name, sae2_name, None, device)
print(model1_name, sae1_name, model2_name, sae2_name)

train_data, _ = load_data(dataset="openwebtext", device=device)

stats = get_running_activation_stats(model1, model2, train_data, batch_size=args.batch_size, n_batches=args.n_batches, seed=34)

torch.save(stats, f"stats_data/stats_{model1_name}_{model2_name}_{sae1_name}_{sae2_name}.pt")