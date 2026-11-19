from pathlib import Path
from typing import Any, Mapping, Optional

from omegaconf import DictConfig, OmegaConf


_UNET_STEM_WEIGHT_SUFFIX = "unet.stem.block.0.weight"


def infer_unet_frame_channels(
    state_dict: Mapping[str, Any], input_timesteps: int
) -> Optional[int]:
    """Infer input channels per timestep from a flattened U-Net checkpoint."""
    if input_timesteps <= 0:
        raise ValueError("input_timesteps must be > 0.")

    weight_keys = [
        key for key in state_dict if key.endswith(_UNET_STEM_WEIGHT_SUFFIX)
    ]
    if not weight_keys:
        return None
    if len(weight_keys) > 1:
        raise ValueError("Checkpoint contains multiple U-Net stem weights.")

    weight = state_dict[weight_keys[0]]
    if len(weight.shape) != 4:
        raise ValueError(
            f"Expected a 4D U-Net stem weight, got shape {tuple(weight.shape)}."
        )

    flattened_channels = int(weight.shape[1])
    if flattened_channels % input_timesteps:
        raise ValueError(
            f"Checkpoint stem has {flattened_channels} channels, which is not "
            f"divisible by input_timesteps={input_timesteps}."
        )
    return flattened_channels // input_timesteps


def dataset_config_for_frame_channels(
    dataset_cfg: DictConfig, frame_channels: int, config_dir: Path
) -> DictConfig:
    """Load a compatible built-in dataset config while preserving active overrides."""
    config_names = {1: "gsmap.yaml", 4: "gsmap_enso.yaml"}
    if frame_channels not in config_names:
        raise ValueError(
            f"No built-in dataset configuration supports {frame_channels} input "
            "channels per timestep."
        )

    if int(dataset_cfg.get("input_channels", 1)) == frame_channels:
        return dataset_cfg

    selected = OmegaConf.load(config_dir / config_names[frame_channels])
    active = OmegaConf.to_container(dataset_cfg, resolve=False)
    selected_data = OmegaConf.to_container(selected, resolve=False)

    for key, value in active.items():
        if key in ("input_channels", "dataset") or key not in selected_data:
            continue
        selected_data[key] = value

    active_dataset = active.get("dataset", {})
    selected_dataset = selected_data["dataset"]
    for key, value in active_dataset.items():
        if key != "_target_" and key in selected_dataset:
            selected_dataset[key] = value

    return OmegaConf.create(selected_data)
