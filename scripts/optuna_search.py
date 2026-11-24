"""Busca de hiperparâmetros (Optuna) e comparação final com/sem ENSO.

Rode na raiz do repositório, com o pacote instalado (uv pip install -e .).

1) Busca SEM ENSO (um processo por GPU; todos compartilham o mesmo estudo):
   CUDA_VISIBLE_DEVICES=0 python scripts/optuna_search.py search \
       --dataset gsmap --model unet/direct --zarr-path /caminho/AINPP.zarr \
       --study unet_noenso --storage optuna.log --n-trials 15

2) Treino final, MESMOS hiperparâmetros, com e sem ENSO, várias sementes:
   python scripts/optuna_search.py final \
       --model unet/direct --zarr-path /caminho/AINPP.zarr \
       --enso-csv ONI_classificado.csv --study unet_noenso --storage optuna.log
"""
import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, get_worker_info

from ainpp_pb_latam.utils import build_loss

ROOT = Path(__file__).resolve().parents[1]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- config
def make_cfg(a, dataset, p):
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
        # training=default: o config.yaml vem com 'gan', mas o loop supervisionado usa 'default'
        cfg = compose(
            config_name="config",
            overrides=[f"model={a.model}", f"dataset={dataset}", "training=default"],
        )
    d = cfg.dataset
    d.dataset.zarr_path = a.zarr_path
    if "enso_csv" in d.dataset:
        d.dataset.enso_csv = a.enso_csv
    d.train_loader.batch_size = p["batch_size"]
    d.train_loader.num_workers = d.val_loader.num_workers = a.workers
    if a.train_steps:
        d.overrides.train.steps_per_epoch = a.train_steps
    if a.val_steps:
        d.overrides.validation.steps_per_epoch = a.val_steps
    cfg.loss.losses[0].alpha = p["alpha"]  # peso para chuva forte (WeightedMSE)
    cfg.loss.weights[1] = p["w_ssim"]  # peso do SSIM
    if "features" in cfg.model:  # largura do U-Net
        base = p.get("width", 64)
        cfg.model.features = [base * 2**i for i in range(4)]
    return cfg


def make_loader(cfg, split, seed):
    d = cfg.dataset
    kw = OmegaConf.to_container(d.overrides[split])
    ds = instantiate(d.dataset, seed=seed, **kw)
    lc = OmegaConf.to_container(d.train_loader if split == "train" else d.val_loader)
    if lc["num_workers"] == 0:
        lc.pop("prefetch_factor", None)

    def winit(wid):  # evita que todos os workers repitam o mesmo rng
        get_worker_info().dataset.rng = np.random.default_rng(seed * 1000 + wid)

    return DataLoader(ds, shuffle=False, worker_init_fn=winit, **lc)


# --------------------------------------------------------------------------- treino
@torch.no_grad()
def evaluate(model, loader, crit, thr=1.0):
    model.eval()
    tot = hit = miss = fa = 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        p = model(x)
        tot += crit(p, y).item()
        pe, ob = p > thr, y > thr
        hit += (pe & ob).sum().item()
        miss += (~pe & ob).sum().item()
        fa += (pe & ~ob).sum().item()
    return tot / len(loader), hit / max(hit + miss + fa, 1)  # (val loss, CSI@thr)


def run(cfg, p, seed, epochs, metric, trial=None, patience=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tl, vl = make_loader(cfg, "train", seed), make_loader(cfg, "validation", seed)
    model = instantiate(cfg.model).to(DEV)
    crit = build_loss(cfg.loss).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["wd"])
    best, bad = {"score": float("inf")}, 0
    try:
        for ep in range(epochs):
            model.train()
            for x, y in tl:
                x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
                opt.zero_grad()
                crit(model(x), y).backward()
                opt.step()
            vloss, csi = evaluate(model, vl, crit)
            score = vloss if metric == "loss" else 1.0 - csi  # sempre minimizar
            if not np.isfinite(score):
                raise optuna.TrialPruned()  # divergiu
            if trial is not None:
                trial.report(score, ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if score < best["score"] - 1e-4:
                best, bad = {"score": score, "val_loss": vloss, "csi": csi, "epoch": ep}, 0
            else:
                bad += 1
                if patience and bad >= patience:
                    break
    finally:
        del model, opt
        torch.cuda.empty_cache()
    return best


# --------------------------------------------------------------------------- optuna
def get_storage(s):
    if s.endswith(".log"):  # vários processos/GPUs sem 'database is locked'
        try:
            return optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(s))
        except AttributeError:  # optuna < 4
            return optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(s))
    return s


def sample(trial, a):
    p = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
        "alpha": trial.suggest_float("alpha", 0.0, 5.0),
        "w_ssim": trial.suggest_float("w_ssim", 0.0, 0.5),
    }
    if a.model.startswith("unet"):
        p["width"] = trial.suggest_categorical("width", [32, 64])
    return p


def cmd_search(a):
    study = optuna.create_study(
        study_name=a.study,
        storage=get_storage(a.storage),
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=a.seed),
        pruner=optuna.pruners.HyperbandPruner(min_resource=2, max_resource=a.epochs),
        load_if_exists=True,
    )

    def objective(trial):
        p = sample(trial, a)
        cfg = make_cfg(a, a.dataset, p)
        return run(cfg, p, a.seed, a.epochs, a.metric, trial)["score"]

    study.optimize(objective, n_trials=a.n_trials)
    print("best:", study.best_value, study.best_params)


def cmd_final(a):
    study = optuna.load_study(study_name=a.study, storage=get_storage(a.storage))
    p = study.best_params
    print("hiperparâmetros:", p)
    out = {}
    for ds in ("gsmap", "gsmap_enso"):
        res = []
        for seed in a.seeds:
            r = run(make_cfg(a, ds, p), p, seed, a.final_epochs, a.metric, patience=10)
            print(ds, seed, r)
            res.append(r["score"])
        out[ds] = {"scores": res, "mean": float(np.mean(res)), "std": float(np.std(res))}
        print(ds, f"{out[ds]['mean']:.4f} ± {out[ds]['std']:.4f}")
    Path(a.out).write_text(json.dumps({"params": p, "results": out}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["search", "final"])
    ap.add_argument("--model", default="unet/direct")
    ap.add_argument("--dataset", default="gsmap")  # só no 'search'
    ap.add_argument("--zarr-path", required=True)
    ap.add_argument("--enso-csv", default=str(ROOT / "ONI_classificado.csv"))
    ap.add_argument("--study", default="nowcast")
    ap.add_argument("--storage", default="optuna.log")
    ap.add_argument("--metric", choices=["loss", "csi"], default="loss")
    ap.add_argument("--n-trials", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=12)  # por trial
    ap.add_argument("--final-epochs", type=int, default=50)
    ap.add_argument("--train-steps", type=int, default=150)  # batches/época (None = config)
    ap.add_argument("--val-steps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="final_enso_vs_noenso.json")
    a = ap.parse_args()
    {"search": cmd_search, "final": cmd_final}[a.mode](a)