# type: ignore
import torch
import plotly.express as px
from univ_utils import load_model_and_sae, plot_scatter
from universality.univ_utils import get_running_activation_stats, load_data

torch.set_grad_enabled(False);

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1_name", type=str, default="out-layer-8_embd-256")
    parser.add_argument("--sae1_name", type=str, default="i510ldxw")
    parser.add_argument("--model2_name", type=str, default="out-layer-8_embd-128")
    parser.add_argument("--sae2_name", type=str, default="i4qnkjqw")
    parser.add_argument("--n_batches", type=int, default=10)
    parser.add_argument("--wandb_log", type=int, default=0)
    parser.add_argument("--flip_models", type=int, default=0)
    args = parser.parse_args()

    if args.flip_models:
        temp = args.model1_name
        args.model1_name = args.model2_name
        args.model2_name = temp

        temp = args.sae1_name
        args.sae1_name = args.sae2_name
        args.sae2_name = temp

    ## for backwards compatibility with old code and easier wandb comparisons
    args.model1_path = args.model1_name
    args.sae1_path = args.sae1_name
    args.model2_path = args.model2_name
    args.sae2_path = args.sae2_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data, val_data = load_data(dataset="openwebtext", device=device)

    w_model, w_sae = load_model_and_sae(model_name=args.model1_name,
                                    sae_name=args.sae1_name,
                                    device=device)
    n_model, n_sae = load_model_and_sae(model_name=args.model2_name,
                                    sae_name=args.sae2_name,
                                    device=device)

    if not args.flip_models:
        assert w_model.cfg.d_model >= n_model.cfg.d_model, "Wide model must have greater or equal embedding dimension to narrow model"

    stats = get_running_activation_stats(w_model, n_model, train_data, batch_size=32, n_batches=80, seed=34)

    print("Computing correlations...")
    corr_matrix = stats.estimate_corr_matrix()

    max_corr = corr_matrix.amax(dim=-1)
    print(f"Mean max correlation: {max_corr.mean()}")

    max_corr_opp = corr_matrix.amax(dim=0)
    print(f"Mean max correlation opposite: {max_corr_opp.mean()}")

    fig1 = plot_scatter(stats.max_x, max_corr, "Wide max acts vs corr", "Max acts", "Corr")
    fig2 = plot_scatter(stats.max_y, max_corr_opp, "Narrow max acts vs reverse corr", "Max acts", "Corr")
    fig3 = px.histogram(max_corr[max_corr > -2].cpu().numpy(), title="Activation Similarity Histogram")
    fig4 = px.histogram(max_corr_opp[max_corr_opp > -2].cpu().numpy(), title="Reverse Activation Similarity Histogram")

    most_similar_n_feats_to_wide = corr_matrix.max(dim=-1).indices
    # most_similar_n_feats_density = n_acts_density[most_similar_n_feats_to_wide]
    # # Do similar features have similar density?
    # fig5 = plot_scatter(w_acts_density, most_similar_n_feats_density, "Wide feature density vs density of similar features in narrow model", "Wide density", "Narrow similar density")

    # Do similar features have similar activation?
    most_similar_n_feats_acts = stats.max_y[most_similar_n_feats_to_wide]
    fig6 = plot_scatter(stats.max_x, most_similar_n_feats_acts, "Wide max acts vs activation of similar features in narrow model", "Wide max acts", "Narrow similar acts")

    #fig7 = plot_scatter(stats.mean_x, max_corr, "Wide mean acts vs corr", "Mean acts", "Corr")
    #fig8 = plot_scatter(stats.mean_y, max_corr_opp, "Narrow mean acts vs reverse corr", "Mean acts", "Corr", z_tensor=stats.mean_y, z_label="Narrow mean")
    # fig9 = plot_scatter(w_nonzero_medians, max_corr, "Wide median acts vs corr", "Median acts", "Corr", z_tensor=w_acts_density, z_label="Wide density")
    # fig10 = plot_scatter(n_nonzero_medians, max_corr_opp, "Narrow median acts vs reverse corr", "Median acts", "Corr", z_tensor=n_acts_density, z_label="Narrow density")

    if args.wandb_log:
        import wandb
        wandb.init(project="sae_universality", config=vars(args))
        wandb.log({"scatter_plot": fig1,
                   "scatter_plot_opp": fig2,
                   "histogram": fig3,
                   "histogram_opp": fig4,
                   # "scatter_plot_density_similar": fig5,
                   "scatter_plot_acts_similar": fig6,
                   # "scatter_plot_mean_acts": fig7,
                   # "scatter_plot_mean_acts_opp": fig8,
                   #"scatter_plot_median_acts": fig9,
                   #"scatter_plot_median_acts_opp": fig10,
                   })


############## -------------- ###################

    # fig9 = px.histogram(w_acts_density, title="Wide feature density histogram")
    # fig10 = px.histogram(n_acts_density, title="Narrow feature density histogram")

    # fig11 = plot_scatter(w_max_acts.cpu() * w_acts_density, max_corr, "Wide max acts * density vs corr", "Max acts * density", "Corr")
    # fig12 = plot_scatter(w_max_acts.cpu() / w_acts_density, max_corr, "Wide max acts / density vs corr", "Max acts / density", "Corr")

    # fig1.show()
    # fig2.show()
    # fig7.show()
    # fig8.show()
    # fig9.show()
    # fig10.show()
    # fig11.show()
    # fig12.show()
