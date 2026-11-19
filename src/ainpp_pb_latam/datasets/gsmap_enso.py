"""
Extensão do AINPPPBLATAMDataset com a fase ENSO (El Niño / La Niña / Neutro)
codificada em one-hot e usada como feature.

Salvar como: ainpp_pb_latam/datasets/gsmap_enso.py

Dois modos de uso (parâmetro `enso_mode`):

* "channels" (padrão): o one-hot é replicado no tempo e no espaço e concatenado
  ao tensor de entrada. x passa de (T, 1, H, W) para (T, 4, H, W).
  Funciona com qualquer backbone convolucional, bastando trocar
  in_channels de 1 para 4.

* "vector": x continua (T, 1, H, W) e o one-hot sai separado, shape (3,).
  Útil se você for condicionar o modelo (concat no bottleneck, FiLM, etc.).
"""

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch

from ainpp_pb_latam.datasets.gsmap import AINPPPBLATAMDataset

# Ordem FIXA das colunas do one-hot. Não use pd.get_dummies: ele só cria
# colunas das categorias presentes, e a ordem/dimensão poderia mudar entre
# train/validation/test.
ENSO_CLASSES: Tuple[str, ...] = ("el_nino", "la_nina", "neutro")
ENSO_TO_CODE: Dict[str, int] = {c: i for i, c in enumerate(ENSO_CLASSES)}


def load_enso_table(csv_path: Union[str, Path]) -> pd.Series:
    """
    Lê o ONI_classificado.csv (colunas: Date, ONI, Classificacao) e devolve uma
    Series indexada por mês (Period[M]) com o código inteiro da fase ENSO.
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df["Classificacao"] = df["Classificacao"].str.strip().str.lower()

    unknown = set(df["Classificacao"]) - set(ENSO_CLASSES)
    if unknown:
        raise ValueError(f"Classes ENSO inesperadas no CSV: {sorted(unknown)}")

    df["period"] = df["Date"].dt.to_period("M")
    if df["period"].duplicated().any():
        raise ValueError("Há meses duplicados no CSV do ONI.")

    return pd.Series(
        df["Classificacao"].map(ENSO_TO_CODE).to_numpy(),
        index=pd.PeriodIndex(df["period"]),
        name="enso_code",
    )


def build_enso_index(time_values: np.ndarray, enso_table: pd.Series) -> np.ndarray:
    """
    Mapeia cada passo de tempo do Zarr para o código ENSO do respectivo mês.
    Retorna um array int64 de shape (T,).
    """
    periods = pd.DatetimeIndex(time_values).to_period("M")
    mapped = enso_table.reindex(periods)
    if mapped.isna().any():
        missing = sorted({str(p) for p in periods[mapped.isna().to_numpy()]})
        raise ValueError(
            f"Meses do Zarr sem classificação ENSO no CSV: {missing[:5]}"
            f"{' ...' if len(missing) > 5 else ''}"
        )
    return mapped.to_numpy(dtype=np.int64)


class AINPPPBLATAMEnsoDataset(AINPPPBLATAMDataset):
    """
    AINPPPBLATAMDataset + fase ENSO em one-hot.

    Args:
        enso_csv: caminho do ONI_classificado.csv.
        enso_mode: "channels" -> x vira (T, 1+3, H, W);
                   "vector"   -> retorna (x, y, onehot[3]).
        enso_reference: instante usado para definir a fase da amostra.
            "last_input"   -> último frame de entrada (momento de emissão da
                              previsão; é o que estaria disponível em
                              operação). Padrão.
            "target_start" -> primeiro frame alvo (t+1).
        Demais argumentos: iguais aos do AINPPPBLATAMDataset.
    """

    def __init__(
        self,
        *args,
        enso_csv: Union[str, Path],
        enso_mode: str = "channels",
        enso_reference: str = "last_input",
        return_metadata: bool = False,
        **kwargs,
    ) -> None:
        if enso_mode not in ("channels", "vector"):
            raise ValueError("enso_mode deve ser 'channels' ou 'vector'.")
        if enso_reference not in ("last_input", "target_start"):
            raise ValueError("enso_reference deve ser 'last_input' ou 'target_start'.")

        # A classe base só devolve t0 via metadata. Forçamos return_metadata=True
        # internamente e respeitamos a escolha do usuário na saída.
        self._user_return_metadata = return_metadata
        super().__init__(*args, return_metadata=True, **kwargs)

        self.enso_mode = enso_mode
        self.enso_reference = enso_reference

        table = load_enso_table(enso_csv)
        self.enso_idx = build_enso_index(self.ds["time"].values, table)  # (T,)
        self._onehot = torch.eye(len(ENSO_CLASSES), dtype=self.dtype)  # (3, 3)

    @property
    def in_channels(self) -> int:
        """Número de canais de entrada que o modelo deve esperar."""
        return 1 + (len(ENSO_CLASSES) if self.enso_mode == "channels" else 0)

    def __getitem__(self, idx: int):
        x, y, meta = super().__getitem__(idx)

        offset = self.input_timesteps - 1 if self.enso_reference == "last_input" else self.input_timesteps
        code = int(self.enso_idx[meta["t0"] + offset])
        onehot = self._onehot[code]  # (3,)

        if self.enso_mode == "channels":
            T, _, H, W = x.shape
            enso_ch = onehot.view(1, -1, 1, 1).expand(T, -1, H, W)
            x = torch.cat([x, enso_ch], dim=1)  # (T, 4, H, W)
            out = (x, y)
        else:
            out = (x, y, onehot.clone())

        if self._user_return_metadata:
            meta["enso"] = ENSO_CLASSES[code]
            return (*out, meta)
        return out
