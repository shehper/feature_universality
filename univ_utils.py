import gc
import os
import torch
from tqdm import tqdm
import plotly.express as px
import numpy as np
from vis_utils import display_vis_inline

from transformer_lens import HookedTransformerConfig
from sae_lens import HookedSAETransformer, SAE

torch.set_grad_enabled(False);
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_nanogpt_sae_transformer(ckpt_path, device):
    # load checkpoint
    try:
        ckpt_file = os.path.join(ckpt_path, "ckpt.pt")
        checkpoint = torch.load(ckpt_file, map_location=device)
    except Exception as e:
        print(f"""Error loading checkpoint: {e}, 
                Expects full path to the model checkpoint as model_name.""")

    # get config
    
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

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def load_model_and_sae(model_path, sae_path, device):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = load_nanogpt_sae_transformer(ckpt_path=os.path.join(parent_dir, model_path), device=device)
    sae_path = os.path.join(parent_dir, "checkpoints", sae_path, "final_1024000000")
    sae = SAE.load_from_pretrained(path=sae_path, device=device)
    sae.fold_W_dec_norm()
    sae.eval()
        
    # splice sae into the model
    hook_name_to_sae = {sae.cfg.hook_name: sae}
    model.add_sae(sae)
    return model, sae

def load_model_and_unnormalized_sae(model_path, sae_path, device):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = load_nanogpt_sae_transformer(ckpt_path=os.path.join(parent_dir, model_path), device=device)
    sae_path = os.path.join(parent_dir, "checkpoints", sae_path, "final_1024000000")
    sae = SAE.load_from_pretrained(path=sae_path, device=device)
    # sae.fold_W_dec_norm()
    sae.eval()
        
    # splice sae into the model
    hook_name_to_sae = {sae.cfg.hook_name: sae}
    model.add_sae(sae)
    return model, sae

@torch.no_grad()
def get_max_and_all_acts(model, sae, data, device=device, n_batches=10, seed=34):
    # TODO: clean up this function

    ## Setting seed so that I compute activations with the same random data for all models
    torch.manual_seed(seed)
    print(f"Computing activations with seed = {seed}")

    sae_acts_hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post" # TODO: 
    activation_cache = {}
    def cache_activations(activation, hook):
        activation_cache[hook.name] = activation.detach()

    all_acts = []
    max_acts = torch.zeros(sae.cfg.d_sae, device=device)

    # Anthropic used 4.1e7 tokens. 4.1e7/(512*32) ~ 2500 batches
    # TODO: do I need to sample activations?
    for _ in tqdm(range(n_batches)): 
        batch_tokens, batch_targets = get_batch(data, 
                                                block_size=sae.cfg.context_size,
                                                batch_size=8)
        out = model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[(sae_acts_hook_name, cache_activations)]
        )
        cached_activations = activation_cache[sae_acts_hook_name]
        batch_max_acts = cached_activations.amax(dim=(0, 1))
        max_acts = torch.maximum(max_acts, batch_max_acts)

        all_acts.append(cached_activations.cpu().view(-1, sae.cfg.d_sae))
    

    del activation_cache; gc.collect(); torch.cuda.empty_cache()
    all_acts = torch.cat(all_acts, dim=0)
    return max_acts, all_acts

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

def get_feature_density(acts, d_sae):
    assert acts.ndim == 2
    assert acts.shape[-1] == d_sae

    density = acts.count_nonzero(dim=0) / acts.shape[0]

    return density