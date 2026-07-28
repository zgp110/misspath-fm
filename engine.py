"""Shared data / training / evaluation plumbing used by train.py and evaluate.py."""

import math
import time

import numpy as np
import torch
import torch.nn as nn

from data import (
    generate_synthetic_data, load_ettm1, load_electricity, load_physionet2012,
    build_dataloaders, build_physionet_dataloaders,
)
from model import MissPathFM

DATASETS = ("ettm1", "synthetic", "electricity", "physionet")
MECHANISMS = ("mcar", "block", "mnar", "irregular")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dataset(name, data_dir, seq_len=48):
    """Returns (data_object, is_physionet)."""
    if name == "ettm1":
        return load_ettm1(data_dir, seq_len=seq_len), False
    if name == "synthetic":
        return generate_synthetic_data(n_samples=2000, seq_len=seq_len, n_features=8), False
    if name == "electricity":
        return load_electricity(data_dir, seq_len=seq_len, n_features=20), False
    if name == "physionet":
        return load_physionet2012(data_dir), True
    raise ValueError(f"Unknown dataset: {name}")


def build_loaders(data_obj, is_physionet, missing_rate, mechanism, batch_size, seed):
    """Returns (train_loader, eval_loader).

    For the three synthetic-missingness datasets the evaluation split holds out a
    fixed missingness pattern (MCAR / block / MNAR / irregular) and MAE is computed
    over the masked-out entries. For PhysioNet 2012 the evaluation split is the
    held-out 10% of patients with an additional artificial hold-out, and MAE is
    computed over `eval_mask` only.
    """
    if is_physionet:
        values, masks = data_obj
        train_loader, _, test_loader = build_physionet_dataloaders(
            values, masks,
            artificial_missing_rate=missing_rate,
            batch_size=batch_size, seed=seed,
        )
        return train_loader, test_loader

    train_loader, val_loader = build_dataloaders(
        data_obj, missing_rate=missing_rate, mechanism=mechanism,
        batch_size=batch_size, seed=seed, random_rate_train=True,
    )
    return train_loader, val_loader


def n_features_of(data_obj, is_physionet):
    return data_obj[0].shape[2] if is_physionet else data_obj.shape[2]


# ---------------------------------------------------------------------------
# Model / checkpoints
# ---------------------------------------------------------------------------

def build_model(config):
    return MissPathFM(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        prior=config["prior"],
        path=config["path"],
        context=config["context"],
    )


def save_checkpoint(path, model, config):
    torch.save({"config": config, "state_dict": model.state_dict()}, path)


def load_checkpoint(path, device):
    """Rebuilds the model from a checkpoint. Returns (model, config)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = build_model(config)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), config


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, train_loader, device, n_epochs=500, lr=1e-3, weight_decay=1e-4,
                warmup=5, eval_every=50, log_fn=print):
    """AdamW + linear warmup + cosine decay, gradient clipping at 1.0.

    Every `eval_every` epochs the training CFM loss is re-measured and the best
    state is kept. Model selection never touches the evaluation split.
    """
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_fn(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, n_epochs - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    best_loss, best_state = float("inf"), None
    t0 = time.time()

    for ep in range(1, n_epochs + 1):
        model.train()
        for batch in train_loader:
            bd = {k: v.to(device) for k, v in batch.items() if k != "eval_mask"}
            opt.zero_grad()
            losses = model.compute_loss(bd)
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if ep % eval_every == 0 or ep == n_epochs:
            model.eval()
            total, nb = 0.0, 0
            with torch.no_grad():
                for batch in train_loader:
                    bd = {k: v.to(device) for k, v in batch.items() if k != "eval_mask"}
                    total += model.compute_loss(bd)["loss"].item()
                    nb += 1
            avg = total / max(1, nb)
            log_fn(f"  ep {ep:4d}/{n_epochs}  loss={avg:.4f}  elapsed={time.time() - t0:.0f}s")
            if avg < best_loss:
                best_loss = avg
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return {"best_train_loss": best_loss, "train_seconds": time.time() - t0}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_imputation(model, loader, device, n_steps=20, n_samples=20, is_physionet=False):
    """Imputation MAE / RMSE over held-out entries, averaged over batches.

    The prediction is the mean of `n_samples` ODE trajectories, each integrated
    with `n_steps` Euler steps (K = 20, Euler-20 in the paper).
    """
    model.eval()
    all_mae, all_rmse = [], []

    for batch in loader:
        x1 = batch["x1"].to(device)
        x_obs = batch["x_obs"].to(device)
        mask = batch["mask"].to(device)
        tg = batch["time_gaps"].to(device)

        pred = torch.stack([model.sample(x_obs, mask, tg, n_steps=n_steps)
                            for _ in range(n_samples)]).mean(0)

        target_mask = batch["eval_mask"].to(device) if is_physionet else (1 - mask)
        denom = target_mask.sum()
        if denom > 0:
            all_mae.append(((target_mask * (pred - x1).abs()).sum() / denom).item())
            all_rmse.append(((target_mask * (pred - x1) ** 2).sum() / denom).sqrt().item())

    return {
        "mae": float(np.mean(all_mae)) if all_mae else float("nan"),
        "rmse": float(np.mean(all_rmse)) if all_rmse else float("nan"),
        "n_batches": len(all_mae),
    }


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(gpu):
    return torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
