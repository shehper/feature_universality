import torch
import os
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"

args = argparse.ArgumentParser()
args.add_argument("--n_layers", type=int, default=8)
args.add_argument("--d_embd", type=int, default=128)
args.add_argument("--hook_layer", type=int, default=6)
args.add_argument("--hook_place", type=str, default="hook_resid_pre")
args.add_argument("--training_steps", type=int, default=250_000)
args.add_argument("--batch_size", type=int, default=4096)
args.add_argument("--expansion_factor", type=int, default=32)
args.add_argument("--architecture", type=str, default="gated")
args.add_argument("--l1_coefficient", type=float, default=5)
args.add_argument("--lr", type=float, default=5e-5)
args.add_argument("--ckpt_iter", type=int, default=None)
args = args.parse_args()

args.device = "cuda" if torch.cuda.is_available() else "cpu"

# extract the model architecture parameters
d_in = args.d_embd * 4 if "mlp" in args.hook_place else args.d_embd
hook_name = f"blocks.{args.hook_layer}.{args.hook_place}"

# load the transformer model
parent_dir = os.path.dirname(os.path.abspath(__file__))
model_dir_name = f"{args.n_layers}-{args.d_embd}"
ckpt_name = f"ckpt_{args.ckpt_iter}" if args.ckpt_iter else "ckpt.pt"
ckpt_path = os.path.join(parent_dir, "llm_ckpts", model_dir_name, ckpt_name)

# set up the training hyperparameters
training_tokens = args.training_steps * args.batch_size
lr_warm_up_steps = 0
lr_decay_steps = args.training_steps // 5  # 20% of training
l1_warm_up_steps = args.training_steps // 20  # 5% of training


# set up the SAE training runner config
cfg = LanguageModelSAERunnerConfig(
    model_class_name="nanogpt",
    architecture=args.architecture,
    model_name=ckpt_path,      
    hook_name=hook_name, 
    hook_layer=args.hook_layer, 
    d_in=d_in,  
    dataset_path="Skylion007/openwebtext", # TODO not sure if this is right
    is_dataset_tokenized=False,
    streaming=True,  
    
    mse_loss_normalization=None, # TODO
    expansion_factor=args.expansion_factor,  
    b_dec_init_method="zeros",  # TODO?
    apply_b_dec_to_input=True, # TODO
    normalize_sae_decoder=False, # TODO
    scale_sparsity_penalty_by_decoder_norm=False, # TODO
    decoder_heuristic_init=True, # TODO
    init_encoder_as_decoder_transpose=True, # TODO
    normalize_activations="expected_average_only_in", # TODO
    
    lr=args.lr,  
    adam_beta1=0.9,  # TODO sweep?
    adam_beta2=0.999,
    lr_scheduler_name="constant",  # TODO sweep?
    lr_warm_up_steps=lr_warm_up_steps,  
    lr_decay_steps=lr_decay_steps,  
    l1_coefficient=args.l1_coefficient,  # TODO
    l1_warm_up_steps=l1_warm_up_steps,  # TODO
    lp_norm=1.0,  # TODO
    train_batch_size_tokens=args.batch_size,
    context_size=512,  # TODO
    n_batches_in_buffer=64,  # TODO
    training_tokens=training_tokens,  # TODO
    store_batch_size_prompts=16, # TODO
    use_ghost_grads=False,  # TODO
    feature_sampling_window=1000,  # TODO
    dead_feature_window=1000,  # TODO
    dead_feature_threshold=1e-4,  # TODO
    log_to_wandb=True,  
    wandb_project="sae_nanogpt",
    wandb_log_frequency=30,
    eval_every_n_wandb_logs=20,
    device=args.device,
    seed=42,
    n_checkpoints=1,
    checkpoint_path="checkpoints",
    dtype="float32",
)

# run the SAE training runner
if __name__ == "__main__":
    sparse_autoencoder = SAETrainingRunner(cfg).run()
