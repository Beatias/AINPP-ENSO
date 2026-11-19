from pathlib import Path

import pytest
from omegaconf import OmegaConf

from ainpp_pb_latam.evaluation.checkpoint import (
    dataset_config_for_frame_channels,
    infer_unet_frame_channels,
)


class FakeWeight:
    def __init__(self, shape):
        self.shape = shape


def test_infer_unet_frame_channels_from_checkpoint():
    state_dict = {"unet.stem.block.0.weight": FakeWeight((64, 48, 3, 3))}

    assert infer_unet_frame_channels(state_dict, input_timesteps=12) == 4


def test_infer_unet_frame_channels_rejects_non_divisible_shape():
    state_dict = {"unet.stem.block.0.weight": FakeWeight((64, 47, 3, 3))}

    with pytest.raises(ValueError, match="not divisible"):
        infer_unet_frame_channels(state_dict, input_timesteps=12)


def test_enso_dataset_selection_preserves_user_patch_sizes():
    config_dir = Path(__file__).resolve().parents[1] / "conf" / "dataset"
    dataset_cfg = OmegaConf.load(config_dir / "gsmap.yaml")
    dataset_cfg.dataset.patch_height = 50
    dataset_cfg.dataset.patch_width = 50

    selected = dataset_config_for_frame_channels(dataset_cfg, 4, config_dir)

    assert selected.input_channels == 4
    assert selected.dataset._target_.endswith("AINPPPBLATAMEnsoDataset")
    assert selected.dataset.patch_height == 50
    assert selected.dataset.patch_width == 50
