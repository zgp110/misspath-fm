"""Evaluate a trained MissPath-FM checkpoint on a held-out imputation split.

Example
-------
    python evaluate.py --ckpt checkpoints/ettm1_mcar_0.3_s42.pt

By default the evaluation setting (dataset / mechanism / missing rate / seed) is
read back from the checkpoint, so the split matches the one used at training
time. Any of those fields can be overridden on the command line to test
transfer to a different missingness setting, e.g.

    python evaluate.py --ckpt checkpoints/ettm1_mcar_0.3_s42.pt \
        --mechanism block --missing_rate 0.5
"""

import argparse
import json
from pathlib import Path

from engine import (
    MECHANISMS, load_dataset, build_loaders, load_checkpoint,
    evaluate_imputation, set_seed, pick_device,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a MissPath-FM checkpoint")
    p.add_argument("--ckpt", type=str, required=True, help="path to a .pt checkpoint")
    p.add_argument("--data_dir", type=str, default="./data")
    # optional overrides of the checkpoint's evaluation setting
    p.add_argument("--mechanism", type=str, default=None, choices=MECHANISMS)
    p.add_argument("--missing_rate", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    # inference
    p.add_argument("--n_steps", type=int, default=20, help="Euler steps")
    p.add_argument("--n_samples", type=int, default=20, help="ODE samples averaged (K)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--result_json", type=str, default=None,
                   help="optional path to write the metrics as JSON")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.gpu)

    model, config = load_checkpoint(args.ckpt, device)

    mechanism = args.mechanism or config["mechanism"]
    missing_rate = args.missing_rate if args.missing_rate is not None else config["missing_rate"]
    seed = args.seed if args.seed is not None else config["seed"]
    batch_size = args.batch_size or config["batch_size"]

    set_seed(seed)
    print(f"[Setup] ckpt={args.ckpt} device={device}", flush=True)
    print(f"[Setup] dataset={config['dataset']} mechanism={mechanism} "
          f"rate={missing_rate} seed={seed}", flush=True)
    print(f"[Model] prior={config['prior']} path={config['path']} "
          f"context={config['context']} hidden={config['hidden_dim']} "
          f"layers={config['n_layers']}", flush=True)

    data_obj, is_physionet = load_dataset(config["dataset"], args.data_dir,
                                          seq_len=config["seq_len"])
    _, eval_loader = build_loaders(data_obj, is_physionet, missing_rate, mechanism,
                                   batch_size, seed)

    metrics = evaluate_imputation(model, eval_loader, device, n_steps=args.n_steps,
                                  n_samples=args.n_samples, is_physionet=is_physionet)
    print(f"[Result] MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
          f"(K={args.n_samples}, Euler-{args.n_steps})", flush=True)

    if args.result_json:
        record = {
            "ckpt": args.ckpt, "dataset": config["dataset"], "mechanism": mechanism,
            "missing_rate": missing_rate, "seed": seed,
            "n_steps": args.n_steps, "n_samples": args.n_samples, **metrics,
        }
        out = Path(args.result_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(record, f, indent=2)
        print(f"[Saved] {out}", flush=True)


if __name__ == "__main__":
    main()
