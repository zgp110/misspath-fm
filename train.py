"""Train MissPath-FM on one (dataset, mechanism, missing rate) setting.

Example
-------
    python train.py --dataset ettm1 --mechanism mcar --missing_rate 0.3

The default hyper-parameters are the ones used for every number reported in the
paper (256 hidden / 4 layers / 500 epochs / random-rate training / K=20 Euler-20).
"""

import argparse
import json
from pathlib import Path

import torch

from engine import (
    DATASETS, MECHANISMS, load_dataset, build_loaders, n_features_of,
    build_model, save_checkpoint, train_model, evaluate_imputation,
    set_seed, pick_device,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train MissPath-FM")
    # setting
    p.add_argument("--dataset", type=str, default="ettm1", choices=DATASETS)
    p.add_argument("--mechanism", type=str, default="mcar", choices=MECHANISMS,
                   help="ignored for --dataset physionet (natural missingness)")
    p.add_argument("--missing_rate", type=float, default=0.3,
                   help="evaluation missing rate; artificial hold-out rate for physionet")
    p.add_argument("--seq_len", type=int, default=48)
    p.add_argument("--data_dir", type=str, default="./data")
    # model
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--prior", type=str, default="interpolation",
                   choices=["interpolation", "gaussian"])
    p.add_argument("--path", type=str, default="missingness_aware",
                   choices=["missingness_aware", "standard"])
    p.add_argument("--context", type=str, default="full",
                   choices=["full", "mask_only", "gap_only", "no_mask"])
    # optimisation
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--eval_every", type=int, default=50,
                   help="epochs between training-loss checks / best-state updates")
    # inference
    p.add_argument("--n_steps", type=int, default=20, help="Euler steps")
    p.add_argument("--n_samples", type=int, default=20, help="ODE samples averaged (K)")
    p.add_argument("--skip_eval", action="store_true", help="train only, no final evaluation")
    # bookkeeping
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument("--result_dir", type=str, default="./results")
    p.add_argument("--tag", type=str, default=None, help="run name (default: derived from setting)")
    return p.parse_args()


def main():
    args = parse_args()
    tag = args.tag or f"{args.dataset}_{args.mechanism}_{args.missing_rate}_s{args.seed}"
    set_seed(args.seed)
    device = pick_device(args.gpu)

    print(f"[Setup] tag={tag} device={device}", flush=True)
    print(f"[Setup] dataset={args.dataset} mechanism={args.mechanism} "
          f"rate={args.missing_rate} seed={args.seed}", flush=True)

    data_obj, is_physionet = load_dataset(args.dataset, args.data_dir, seq_len=args.seq_len)
    train_loader, eval_loader = build_loaders(
        data_obj, is_physionet, args.missing_rate, args.mechanism, args.batch_size, args.seed)

    config = {
        "input_dim": n_features_of(data_obj, is_physionet),
        "hidden_dim": args.hidden_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "prior": args.prior,
        "path": args.path,
        "context": args.context,
        # kept so evaluate.py can rebuild the identical evaluation split
        "dataset": args.dataset,
        "mechanism": args.mechanism,
        "missing_rate": args.missing_rate,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "epochs": args.epochs,
    }

    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] prior={args.prior} path={args.path} context={args.context} "
          f"params={n_params:,}", flush=True)

    stats = train_model(
        model, train_loader, device,
        n_epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        warmup=args.warmup, eval_every=args.eval_every,
        log_fn=lambda m: print(m, flush=True),
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{tag}.pt"
    save_checkpoint(ckpt_path, model, config)
    print(f"[Checkpoint] {ckpt_path}", flush=True)

    record = {"tag": tag, "n_params": n_params, **{k: v for k, v in vars(args).items()}, **stats}

    if not args.skip_eval:
        metrics = evaluate_imputation(
            model, eval_loader, device,
            n_steps=args.n_steps, n_samples=args.n_samples, is_physionet=is_physionet)
        record.update(metrics)
        print(f"[Result] MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}", flush=True)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{tag}.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[Saved] {result_path}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
