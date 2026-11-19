from .gsmap import AINPPPBLATAMDataset
from .gsmap_enso import AINPPPBLATAMEnsoDataset
# Backwards compatibility: older scripts reference NowcastingDataset.
# Keep a thin alias so legacy entry points continue to work.
NowcastingDataset = AINPPPBLATAMDataset

__all__ = [
    "AINPPPBLATAMDataset",
    "AINPPPBLATAMEnsoDataset",
    "NowcastingDataset",
]
    