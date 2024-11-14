# type: ignore
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner


def get_sae_vis_data(sae, model, data, features, device):

    feature_vis_config_gpt = SaeVisConfig(
        hook_point=sae.cfg.hook_name,
        features=features,
        minibatch_size_features=64,
        minibatch_size_tokens=256,
        verbose=True,
        device=device,
    )

    visualization_data_gpt = SaeVisRunner(feature_vis_config_gpt).run(
        encoder=sae,  # type: ignore
        model=model,
        tokens=data,  # type: ignore
    )

    return visualization_data_gpt