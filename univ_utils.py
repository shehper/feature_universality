# type: ignore
import gc
import os
import torch
import umap
import torch
import pandas as pd
from tqdm import tqdm
import plotly.express as px
import numpy as np
import functools
from transformer_lens import HookedTransformerConfig
from sae_lens import HookedSAETransformer, SAE
from running_statistics import RunningStats
from typing import Tuple
torch.set_grad_enabled(False);

# TODO: replace this with huggingface datasets instead.
def load_data(dataset):
    data_parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(data_parent_dir, "data", dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    return train_data, val_data


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def load_model(model_name: str, device: str, ckpt_iter: int) -> HookedSAETransformer:
    # load checkpoint
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(parent_dir, "llm_checkpoints", model_name)
    ckpt_iter_str = "final" if ckpt_iter is None else f"{ckpt_iter}"
    try:
        ckpt_file = os.path.join(ckpt_path, f"ckpt_{ckpt_iter_str}.pt")
        checkpoint = torch.load(ckpt_file, map_location=device)
    except Exception as e:
        print(f"""Error loading checkpoint: {e}, 
                Expects full path to the model checkpoint as model_name.""")

    model_config = checkpoint['model_args']
    cfg = HookedTransformerConfig(
        n_layers=model_config["n_layer"],
        d_model=model_config["n_embd"],
        d_head=int(model_config["n_embd"]/ model_config["n_head"]),
        n_heads=model_config["n_head"],
        d_mlp=model_config["n_embd"] * 4,
        d_vocab=model_config["vocab_size"],
        n_ctx=model_config["block_size"],
        act_fn="gelu",
        normalization_type="LN",
        )

    # load state dict
    from transformer_lens.pretrained.weight_conversions.nanogpt import convert_nanogpt_weights
    state_dict = checkpoint['model']
    new_state_dict = convert_nanogpt_weights(old_state_dict=state_dict, cfg=cfg)
    model = HookedSAETransformer(cfg)
    model.load_state_dict(new_state_dict, strict=False)

    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.set_tokenizer(tokenizer)

    return model

def load_sae(sae_name: str, device: str, ckpt_iter: int, model_name: str, fold_W_dec_norm: bool = True) -> SAE:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_iter_str = "final" if ckpt_iter is None else f"{ckpt_iter}"
    sae_path = os.path.join(parent_dir, "sae_checkpoints", f"{model_name}-{ckpt_iter_str}", sae_name, "final_1024000000")
    sae = SAE.load_from_pretrained(path=sae_path, device=device)
    sae.eval()
    if fold_W_dec_norm:
        sae.fold_W_dec_norm()
    return sae


def load_model_and_sae(model_name: str, 
                       sae_name: str, 
                       ckpt_iter: int,
                       device: str, 
                       fold_W_dec_norm: bool = True, 
                       splice_sae: bool = True) -> Tuple[HookedSAETransformer, SAE]:
    model = load_model(model_name=model_name, device=device, ckpt_iter=ckpt_iter)
    sae = load_sae(sae_name=sae_name, device=device, ckpt_iter=ckpt_iter, model_name=model_name, fold_W_dec_norm=fold_W_dec_norm)
    if splice_sae:
        # splice sae into the model
        hook_name_to_sae = {sae.cfg.hook_name: sae}
        model.add_sae(sae)        
    return model, sae

@torch.no_grad()
def get_running_activation_stats(model1, model2, data, batch_size=32, n_batches=80, seed=34):

    torch.manual_seed(seed)

    assert model1.acts_to_saes.keys() == model2.acts_to_saes.keys(), "models must have the same hooks"
    assert len(model1.acts_to_saes) == 1, "assuming only one hook for now"
    assert model1.cfg.device == model2.cfg.device, "models must be on the same device"

    sae_hook_place = list(model1.acts_to_saes.keys())[0]
    n_latents1 = model1.acts_to_saes[sae_hook_place].cfg.d_sae
    n_latents2 = model2.acts_to_saes[sae_hook_place].cfg.d_sae
    context_size = model1.acts_to_saes[sae_hook_place].cfg.context_size

    hook_name = f"{sae_hook_place}.hook_sae_acts_post"

    cache = {}
    def cache_acts(act, hook, model_name=""):
        act_dim = act.shape[-1]
        if model_name not in cache:
            cache[model_name] = {}
        cache[model_name][hook.name] = act.detach().view(-1, act_dim)

    running_stats = RunningStats(shape=(n_latents1, n_latents2), device=model1.cfg.device)
    
    for _ in tqdm(range(n_batches)): 
        batch_tokens, _ = get_batch(data, 
                                    block_size=context_size,
                                    batch_size=batch_size,
                                    device=model1.cfg.device)

        model1.run_with_hooks(
            batch_tokens,
            fwd_hooks=[(hook_name, functools.partial(cache_acts, model_name="model1"))]
        )

        model2.run_with_hooks(
            batch_tokens,
            return_type=None,
            fwd_hooks=[(hook_name, functools.partial(cache_acts, model_name="model2"))]
        )

        running_stats.update(x=cache["model1"][hook_name],
                             y=cache["model2"][hook_name])
        
    del cache
    gc.collect()
    torch.cuda.empty_cache()
    
    return running_stats

def plot_scatter(x_tensor, y_tensor, title="Scatter Plot", x_label="X", y_label="Y", z_tensor=None, z_label="Z"):
    # TODO: clean up this function
    c = torch.stack((x_tensor.cpu(), y_tensor.cpu()), dim=0)
    title = title + f"\nCorrelation: {torch.corrcoef(c)[0, 1]:.2f}"

    # Create scatter plot with color dimension if z_tensor is provided
    if z_tensor is not None:
        fig = px.scatter(
            x=x_tensor.cpu().numpy(),
            y=y_tensor.cpu().numpy(),
            color=z_tensor.cpu().numpy(),  # Use z_tensor for colors
            color_continuous_scale="viridis",  # High-contrast color map
            range_color=(0, 0.02),  # Set the color range explicitly
            title=title,
            labels={'x': x_label, 'y': y_label, 'color': z_label},  # Add label for colorbar
        )
    else:
        fig = px.scatter(
            x=x_tensor.cpu().numpy(),
            y=y_tensor.cpu().numpy(),
            title=title,
            labels={'x': x_label, 'y': y_label}
        )
    
    # Update layout with titles and axes labels
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(range=[0, x_tensor.max().cpu().numpy()]),
        yaxis=dict(range=[0, y_tensor.max().cpu().numpy()]),
        width=800,
        height=600
    )

    return fig


def get_umap_embeddings(data, n_neighbors=15, min_dist=0.1, random_state=42):
    
    # Initialize UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    
    # Fit and transform the data
    embedding = reducer.fit_transform(data)
    
    return embedding

def create_umap_visualization_from_data(data, bool_labels=None, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Create an interactive 2D UMAP visualization of high-dimensional vectors using Plotly.
    
    Parameters:
    data: numpy array of shape (n_samples, n_features)
    bool_labels: torch.Tensor of shape (n_samples,) with dtype torch.bool
    n_neighbors: int, number of neighbors to consider for manifold structure
    min_dist: float, minimum distance between points in low dimensional representation
    random_state: int, random seed for reproducibility
    
    Returns:
    embedding: numpy array of shape (n_samples, 2)
    fig: plotly figure object
    """
    # Convert boolean tensor to numpy if needed
    if bool_labels is None:
        bool_labels = torch.zeros(data.shape[0], dtype=torch.bool)

    if isinstance(bool_labels, torch.Tensor):
        bool_labels = bool_labels.cpu().numpy()
    
    embedding = get_umap_embeddings(data, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    
    df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'Label': ['True' if x else 'False' for x in bool_labels]
    })
    
    # Create the interactive plot
    fig = px.scatter(
        df,
        x='UMAP1',
        y='UMAP2',
        color='Label',
        color_discrete_map={'True': '#ff7f7f', 'False': '#7f7fff'},
        title='UMAP projection colored by boolean values',
        opacity=0.7,
        hover_data={'UMAP1': ':.2f', 'UMAP2': ':.2f'},
        category_orders={'Label': ['False', 'True']}  # Ensure consistent legend order
    )
    
    # Update layout for better appearance
    fig.update_layout(
        plot_bgcolor='white',
        width=900,
        height=700,
        legend_title_text='Labels',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Update traces for better appearance
    fig.update_traces(
        marker=dict(size=6),
        selector=dict(mode='markers')
    )
    
    return embedding, fig

# Example usage:
# embedding, fig = create_umap_visualization(vectors, bool_labels)
# fig.show()  # This will display the interactive plot in a notebook or browser

# Optional: Save the plot to HTML
# fig.write_html("umap_visualization.html")