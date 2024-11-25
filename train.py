"""
Main script for training a sparse autoencoder on a language model.
Heavily based on the SAE Lens codebase.
"""

import torch
import os
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"

args = argparse.ArgumentParser()
args.add_argument("--n_layers", type=int, default=8)
args.add_argument("--n_embd", type=int, default=128)
args.add_argument("--hook_name", type=str, default="blocks.6.hook_resid_pre")
args.add_argument("--training_steps", type=int, default=250_000)
args.add_argument("--batch_size", type=int, default=4096)
args.add_argument("--expansion_factor", type=int, default=32)
args.add_argument("--l1_coefficient", type=float, default=5)
args.add_argument("--lr", type=float, default=5e-5)
args.add_argument("--ckpt_iter", type=int, default=None)
args = args.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# extract the model architecture parameters
d_in = args.n_embd * 4 if "mlp" in args.hook_name else args.n_embd

# load the transformer model
parent_dir = os.path.dirname(os.path.abspath(__file__))
model_dir_name = f"{args.n_layers}-{args.n_embd}"
ckpt_iter_str = "final" if args.ckpt_iter is None or args.ckpt_iter == int(2.5e5) else f"{args.ckpt_iter}"
ckpt_path = os.path.join(parent_dir, "llm_checkpoints", model_dir_name, f"ckpt_{ckpt_iter_str}.pt")

# set up the training hyperparameters
training_tokens = args.training_steps * args.batch_size
lr_warm_up_steps = 0
lr_decay_steps = args.training_steps // 5  # 20% of training
l1_warm_up_steps = args.training_steps // 20  # 5% of training

# The following dataset is tokenized in the same way as nanoGPT tokenizer
# See https://huggingface.co/datasets/apollo-research/Skylion007-openwebtext-tokenizer-gpt2/blob/
# f02886b54795e8acabceb637ca119f9ae8f19d3f/upload_script.py#L77
# and 
# https://github.com/shehper/nanogpt/blob/ecbaeb246327c3473a07173e06832815b3eb2325/data/openwebtext/prepare.py#L41
# for comparison.
dataset_path = "apollo-research/Skylion007-openwebtext-tokenizer-gpt2"

# checkpoint path
checkpoint_path = os.path.join(parent_dir, "sae_checkpoints", f"{model_dir_name}-{ckpt_iter_str}")

# set up the SAE training runner config
cfg = LanguageModelSAERunnerConfig(
    
    ## Model architecture
    model_class_name="nanogpt",
    model_name=ckpt_path,      
    hook_name=args.hook_name, 
    hook_layer=int(args.hook_name.split(".")[1]), 
    d_in=d_in,
    architecture="gated",

    ## Dataset
    dataset_path=dataset_path,
    is_dataset_tokenized=True,
    streaming=False,  # we don't stream as we downloaded the dataset in advance.
    
    ## SAE training configuration
    mse_loss_normalization=None, # whether to normalize MSE loss
    expansion_factor=args.expansion_factor,  
    b_dec_init_method="zeros", # other methods are "geometric_median" and "mean"
    apply_b_dec_to_input=True, 
    normalize_sae_decoder=False, # whether each decoder column is normalized to unit norm
    scale_sparsity_penalty_by_decoder_norm=False, # whether to scale the sparsity penalty by the decoder norm
    decoder_heuristic_init=True, # whether W_dec is initialized to have columns point in random directions and 
                                 # fixed L2 norm of 0.05 to 1 (as per Anthropic April Update)
    init_encoder_as_decoder_transpose=True, # whether to initialize the encoder as the transpose of the decoder
    normalize_activations="expected_average_only_in", # estimate average activation norm once in the beginning and divide activations by it
    
    ## SAE training hyperparameters
    # learning rate
    lr=args.lr,  
    adam_beta1=0.9, 
    adam_beta2=0.999,
    lr_scheduler_name="constant",  
    lr_warm_up_steps=lr_warm_up_steps,  
    lr_decay_steps=lr_decay_steps,  

    ## Loss function
    l1_coefficient=args.l1_coefficient, 
    l1_warm_up_steps=l1_warm_up_steps,  
    lp_norm=1.0,  # which L_p norm to use for the sparsity penalty

    ## Training configuration
    train_batch_size_tokens=args.batch_size,
    context_size=512,  # TODO: Maybe I can increase it to 1024 to get best performance, thought it might be slow
    n_batches_in_buffer=64, 
    training_tokens=training_tokens,  
    store_batch_size_prompts=16, # number of prompts in a single batch of language model activations
    
    ## Sparsity / Dead Feature Handling
    use_ghost_grads=False,
    feature_sampling_window=1000,  
    dead_feature_window=1000,  
    dead_feature_threshold=1e-4, 
    
    ## Logging and Saving
    log_to_wandb=True,  
    wandb_project="sae_universality",
    wandb_log_frequency=30,
    eval_every_n_wandb_logs=20,
    device=device,
    seed=42,
    n_checkpoints=1,
    checkpoint_path=checkpoint_path,
    dtype="float32",
)

# run the SAE training runner
if __name__ == "__main__":
    sparse_autoencoder = SAETrainingRunner(cfg).run()
