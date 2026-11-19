from .gsmap import AINPPPBLATAMDataset
from .gsmap_enso import AINPPPBLATAMDataset
# Backwards compatibility: older scripts reference NowcastingDataset.
# Keep a thin alias so legacy entry points continue to work.
NowcastingDataset = AINPPPBLATAMDataset
    