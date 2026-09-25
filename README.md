# MissPath-FM: Missingness-Aware Probability Paths for Flow Matching on Partially Observed Time Series

Anonymous code release accompanying our NeurIPS submission on flow matching for time
series with missing values and irregular sampling. It contains the model and the
imputation experiments needed to reproduce the central results of the paper. No
pre-trained weights are shipped — every number is reproduced by training from scratch
with the scripts below.

## Overview

Time series with missing values, irregular sampling, and asynchronous multi-channel
observations are pervasive in healthcare, industrial sensing, and IoT. Existing
flow-matching methods for time series treat missingness primarily as a *conditioning
input* to the velocity backbone — the source distribution and the probability path
itself remain agnostic to where data is observed. **MissPath-FM** instead pushes the
observation pattern into the path itself.

The model has three observation-aware components:

1. **Observation-aware source prior** `p_0(z_0 | c)`. Rather than starting from a fixed
   standard Gaussian, we interpolate the observed values along the time axis and add
   observation-dependent noise: small variance where the value is observed, larger
   variance over the gaps. The source already lies close to the data manifold.
2. **Missingness-aware probability path.** The standard linear path
   `x_t = (1-t) z_0 + t x_1` is replaced by a smooth, speed-modulated path
   `x_t = (1 - alpha_t) z_0 + alpha_t x_1` with `alpha_t = 1 - (1-t)^s(c)`. The
   per-element speed `s(c)` depends on the mask, so observed positions converge faster
   while missing positions retain more transport time.
3. **Conditional velocity field.** A Transformer velocity model takes
   `(x_t, t, mask, time_gaps)` and is trained with the CFM regression target
   `s(c) (x_1 - z_0)`, plus a small consistency penalty at observed positions.

The combination yields shorter, more consistent transport paths whose endpoints respect
the observed data, and a velocity field that is easier to fit on partially observed data.

## Repository layout

```
.
├── model.py            MissPath-FM: source priors, probability paths, velocity field
├── data.py             Datasets (ETTm1, Electricity, Synthetic, PhysioNet 2012) and
│                       missingness mechanisms (MCAR, block, MNAR, irregular)
├── engine.py           Shared loaders, training loop, evaluation, checkpoint I/O
├── train.py            Training entry point (train → checkpoint → evaluate)
├── evaluate.py         Test entry point (load a checkpoint → evaluate)
├── requirements.txt
├── LICENSE
└── scripts/
    ├── smoke_test.sh       ~1 min end-to-end check, no external data needed
    ├── run_ettm1.sh        all 8 ETTm1 settings
    ├── run_synthetic.sh    all 8 Synthetic settings
    ├── run_electricity.sh  all 8 Electricity settings
    ├── run_physionet.sh    all 4 PhysioNet settings
    ├── run_main_table.sh   the full main table, sequentially
    └── run_ablation.sh     prior / path / context ablation on one cell
```

## Setup

```bash
pip install -r requirements.txt     # torch >= 2.0, numpy, pandas
bash scripts/smoke_test.sh 0        # sanity check on GPU 0 (also runs on CPU)
```

The smoke test trains a small model on the synthetic dataset for 10 epochs, writes a
checkpoint, reloads it in `evaluate.py`, and prints MAE / RMSE. It needs no downloaded
data.

## Datasets

| Dataset        | Domain     | Shape         | Missingness in evaluation              |
|----------------|------------|---------------|----------------------------------------|
| ETTm1          | Energy     | (N, 48, 7)    | Synthetic MCAR / block / MNAR / irreg. |
| Electricity    | Energy     | (N, 48, 20)   | Synthetic MCAR / block / MNAR / irreg. |
| Synthetic      | Sinusoidal | (2000, 48, 8) | Synthetic MCAR / block / MNAR / irreg. |
| PhysioNet 2012 | Clinical   | (N, 48, 35)   | Natural + artificial hold-out          |

The synthetic dataset is generated in-process. The other three expect raw files under
`./data/` (override with `--data_dir`):

```
data/
  ETTm1.csv
  electricity.txt
  set-a/                  PhysioNet 2012 set-a, extracted
  set-a.tar.gz            alternative; auto-extracted on first run
```

Sources:
- ETTm1: <https://github.com/zhouhaoyi/ETDataset>
- Electricity (UCI ElectricityLoadDiagrams20112014): <https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014>
- PhysioNet Challenge 2012: <https://physionet.org/content/challenge-2012/1.0.0/>

Every channel is z-normalised. Electricity keeps the 20 highest-variance channels and
subsamples to 3000 windows; PhysioNet is binned into 48 hourly steps over 35 clinical
variables. For PhysioNet the evaluation entries are held out uniformly at random among
the naturally observed values, so `--mechanism` has no effect there and only
`--missing_rate` matters.

## Training

```bash
python train.py --dataset ettm1       --mechanism mcar  --missing_rate 0.3
python train.py --dataset synthetic   --mechanism block --missing_rate 0.5
python train.py --dataset electricity --mechanism mnar  --missing_rate 0.3
python train.py --dataset physionet                     --missing_rate 0.1
```

Each run prints the training loss periodically, writes `checkpoints/<tag>.pt` and
`results/<tag>.json`, and reports the final imputation MAE / RMSE over the held-out
entries. `<tag>` defaults to `<dataset>_<mechanism>_<rate>_s<seed>` and can be set with
`--tag`.

Training draws a random missing rate per sample (`U[0.1, 0.7]`, MCAR) so a single model
covers all evaluation rates; evaluation always uses the fixed mechanism and rate given on
the command line.

## Testing a checkpoint

```bash
python evaluate.py --ckpt checkpoints/ettm1_mcar_0.3_s42.pt
```

The evaluation setting is read back from the checkpoint, so the held-out split matches
the one used during training. Override any field to test transfer to a different
missingness pattern:

```bash
python evaluate.py --ckpt checkpoints/ettm1_mcar_0.3_s42.pt \
    --mechanism block --missing_rate 0.5 --n_samples 20 --n_steps 20
```

## Reproducing the main table

```bash
bash scripts/run_ettm1.sh        0      # GPU 0, all 8 ETTm1 settings
bash scripts/run_synthetic.sh    0
bash scripts/run_electricity.sh  0
bash scripts/run_physionet.sh    0

bash scripts/run_main_table.sh   0      # everything above, sequentially
```

That is 3 datasets × 4 mechanisms × 2 rates, plus PhysioNet at four artificial hold-out
rates {0.1, 0.2, 0.3, 0.5}. Reference wall-clock per setting at the default
configuration on a single 24 GB GPU: ETTm1 ≈ 6 min, Synthetic ≈ 8 min,
Electricity ≈ 3 min, PhysioNet ≈ 32 min.

## Ablations

The observation-aware components can be switched off individually:

| Flag                  | Effect                                                       |
|-----------------------|--------------------------------------------------------------|
| `--prior gaussian`    | standard `N(0, I)` source instead of the interpolation prior |
| `--path standard`     | standard linear path instead of the speed-modulated path     |
| `--context mask_only` | velocity sees the mask but not the time gaps                 |
| `--context gap_only`  | velocity sees the time gaps but not the mask                 |
| `--context no_mask`   | velocity sees only `x_t`                                     |

```bash
bash scripts/run_ablation.sh 0 ettm1 mcar 0.3
```

## Default configuration

These defaults are the configuration behind every number reported in the paper.

| Group     | Setting                                                             |
|-----------|---------------------------------------------------------------------|
| Model     | 256 hidden, 4 layers, 4 heads, FFN 1024, dropout 0.1 (≈3.9M params) |
| Prior     | interpolation, `sigma_obs = 0.1`, `sigma_miss = 1.0`                |
| Path      | missingness-aware, `speed_obs = 2.0`, `speed_miss = 1.0`            |
| Loss      | CFM + 0.1 × observation-consistency penalty                         |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4, grad clip 1.0                    |
| Schedule  | 500 epochs, 5-epoch linear warmup + cosine decay                    |
| Batch     | 64                                                                  |
| Training  | random missing rate `U[0.1, 0.7]` MCAR per sample                   |
| Inference | mean of K = 20 ODE samples, Euler with 20 steps                     |

Model selection keeps the best state by *training* CFM loss (checked every 50 epochs);
the evaluation split is never used for selection.

