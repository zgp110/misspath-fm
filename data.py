import numpy as np
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


def generate_mcar_mask(shape, missing_rate, rng):
    return (rng.random(shape) > missing_rate).astype(np.float32)


def generate_block_mask(shape, missing_rate, rng, min_block_len=5, max_block_len=20):
    T, D = shape
    mask = np.ones(shape, dtype=np.float32)
    target = int(T * D * missing_rate)
    cur = 0
    while cur < target:
        d = rng.integers(0, D)
        L = rng.integers(min_block_len, max_block_len + 1)
        s = rng.integers(0, max(1, T - L))
        e = min(s + L, T)
        cur += int(mask[s:e, d].sum())
        mask[s:e, d] = 0.0
    return mask


def generate_irregular_mask(shape, missing_rate, rng, min_interval=1, max_interval=8):
    T, D = shape
    mask = np.zeros(shape, dtype=np.float32)
    for d in range(D):
        base = rng.integers(min_interval, max_interval + 1)
        t = 0
        while t < T:
            mask[t, d] = 1.0
            t += max(1, base + rng.integers(-2, 3))
        if mask[:, d].mean() > (1 - missing_rate):
            obs_idx = np.where(mask[:, d] > 0.5)[0]
            n_remove = int(len(obs_idx) - T * (1 - missing_rate))
            if n_remove > 0:
                rm = rng.choice(obs_idx, size=min(n_remove, len(obs_idx) - 1), replace=False)
                mask[rm, d] = 0.0
    return mask


def generate_mnar_mask(data, missing_rate, rng, quantile_threshold=0.8, mnar_fraction=0.5):
    T, D = data.shape
    mask = np.ones((T, D), dtype=np.float32)
    total = int(T * D * missing_rate)
    n_mnar = int(total * mnar_fraction)
    n_mcar = total - n_mnar

    abs_data = np.abs(data)
    th = np.quantile(abs_data, quantile_threshold, axis=0)
    high = abs_data > th[None, :]
    high_pos = list(zip(*np.where(high)))
    rng.shuffle(high_pos)

    dropped = 0
    for t, d in high_pos:
        if dropped >= n_mnar:
            break
        if mask[t, d] > 0.5:
            mask[t, d] = 0.0
            dropped += 1

    remaining = n_mcar + (n_mnar - dropped)
    obs_pos = list(zip(*np.where(mask > 0.5)))
    rng.shuffle(obs_pos)
    for t, d in obs_pos[:remaining]:
        mask[t, d] = 0.0
    return mask


def apply_missingness(data, missing_rate, mechanism="mcar", seed=42):
    rng = np.random.default_rng(seed)
    if mechanism == "mcar":
        m = generate_mcar_mask(data.shape, missing_rate, rng)
    elif mechanism == "block":
        m = generate_block_mask(data.shape, missing_rate, rng)
    elif mechanism == "irregular":
        m = generate_irregular_mask(data.shape, missing_rate, rng)
    elif mechanism == "mnar":
        m = generate_mnar_mask(data, missing_rate, rng)
    else:
        raise ValueError(mechanism)
    return data * m, m


def compute_time_gaps(mask):
    T, D = mask.shape
    gaps = np.zeros_like(mask)
    for d in range(D):
        last = -1
        for t in range(T):
            if mask[t, d] > 0.5:
                last = t
                gaps[t, d] = 0.0
            else:
                gaps[t, d] = (t - last) if last >= 0 else (t + 1)
    return gaps.astype(np.float32)


def generate_synthetic_data(n_samples=2000, seq_len=48, n_features=8, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, seq_len)
    out = []
    for _ in range(n_samples):
        f = rng.uniform(0.5, 2.0, size=n_features)
        p = rng.uniform(0, 2 * np.pi, size=n_features)
        a = rng.uniform(0.5, 2.0, size=n_features)
        s = a[None, :] * np.sin(f[None, :] * t[:, None] + p[None, :])
        s += rng.normal(0, 0.1, s.shape)
        out.append(s)
    return np.stack(out).astype(np.float32)


def load_ettm1(data_dir, seq_len=48):
    p = Path(data_dir) / "ETTm1.csv"
    df = pd.read_csv(p)
    v = df.iloc[:, 1:].values.astype(np.float32)
    v = (v - v.mean(0, keepdims=True)) / (v.std(0, keepdims=True) + 1e-8)
    n = len(v) // seq_len
    return v[:n * seq_len].reshape(n, seq_len, -1)


def load_electricity(data_dir, seq_len=48, n_features=20, max_samples=3000):
    p = Path(data_dir) / "electricity.txt"
    v = np.loadtxt(str(p), delimiter=",")
    if v.shape[1] > n_features:
        var = np.nanvar(v, axis=0)
        idx = np.argsort(var)[-n_features:]
        v = v[:, idx]
    v = np.nan_to_num(v, nan=0.0)
    v = (v - v.mean(0, keepdims=True)) / (v.std(0, keepdims=True) + 1e-8)
    n = len(v) // seq_len
    w = v[:n * seq_len].reshape(n, seq_len, -1).astype(np.float32)
    if len(w) > max_samples:
        rng = np.random.default_rng(42)
        w = w[rng.choice(len(w), max_samples, replace=False)]
    return w


PHYSIONET_VARIABLES = [
    "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT",
    "HR", "K", "Lactate", "Mg", "MAP", "MechVent", "Na", "NIDiasABP",
    "NIMAP", "NISysABP", "PaCO2", "PaO2", "pH", "Platelets", "RespRate",
    "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC",
]
PHYSIONET_STATIC = ["RecordID", "Age", "Gender", "Height", "ICUType", "Weight"]


def _parse_physionet_patient(filepath):
    n_vars = len(PHYSIONET_VARIABLES)
    var_to_idx = {v: i for i, v in enumerate(PHYSIONET_VARIABLES)}
    values = np.full((48, n_vars), np.nan, dtype=np.float32)
    with open(filepath, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) != 3:
            continue
        time_str, param, val_str = parts
        if param in PHYSIONET_STATIC or param not in var_to_idx:
            continue
        try:
            val = float(val_str)
        except ValueError:
            continue
        h, _ = time_str.split(":")
        hour = int(h)
        if hour >= 48:
            continue
        values[hour, var_to_idx[param]] = val
    mask = (~np.isnan(values)).astype(np.float32)
    return values, mask


def load_physionet2012(data_dir, subset="set-a"):
    import tarfile
    data_path = Path(data_dir)
    subset_dir = data_path / subset
    tar_path = data_path / f"{subset}.tar.gz"
    if not subset_dir.exists() and tar_path.exists():
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_path)
    files = sorted(subset_dir.glob("*.txt"))
    all_v, all_m = [], []
    for pf in files:
        v, m = _parse_physionet_patient(pf)
        all_v.append(v)
        all_m.append(m)
    values = np.stack(all_v)
    masks = np.stack(all_m)
    for d in range(values.shape[2]):
        obs = values[:, :, d][masks[:, :, d] > 0.5]
        if len(obs) > 0:
            mu, sd = obs.mean(), obs.std() + 1e-8
            values[:, :, d] = np.where(masks[:, :, d] > 0.5, (values[:, :, d] - mu) / sd, 0.0)
    return values, masks


class RandomRateDataset(Dataset):
    def __init__(self, data, rate_low=0.1, rate_high=0.7, seed=42):
        self.data = data.astype(np.float32)
        self.rate_low = rate_low
        self.rate_high = rate_high
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x1 = self.data[idx]
        T, D = x1.shape
        r = self.rng.uniform(self.rate_low, self.rate_high)
        rng_i = np.random.default_rng(self.rng.integers(1e9))
        mask = generate_mcar_mask((T, D), r, rng_i)
        x_obs = x1 * mask
        tg = compute_time_gaps(mask)
        return {
            "x1": torch.from_numpy(x1),
            "x_obs": torch.from_numpy(x_obs),
            "mask": torch.from_numpy(mask),
            "time_gaps": torch.from_numpy(tg),
        }


class FixedMissDataset(Dataset):
    def __init__(self, data, missing_rate=0.3, mechanism="mcar", seed=42):
        self.data = data.astype(np.float32)
        self.missing_rate = missing_rate
        self.mechanism = mechanism
        self.seed = seed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x1 = self.data[idx]
        x_obs, mask = apply_missingness(x1, self.missing_rate, self.mechanism, seed=self.seed + idx)
        tg = compute_time_gaps(mask)
        return {
            "x1": torch.from_numpy(x1),
            "x_obs": torch.from_numpy(x_obs.astype(np.float32)),
            "mask": torch.from_numpy(mask),
            "time_gaps": torch.from_numpy(tg),
        }


class PhysioNetDataset(Dataset):
    def __init__(self, values, natural_masks, artificial_missing_rate=0.1, seed=42):
        self.values = values
        self.natural_masks = natural_masks
        self.artificial_missing_rate = artificial_missing_rate
        self.seed = seed
        self.targets = self._forward_fill(values, natural_masks)

    def _forward_fill(self, values, masks):
        N, T, D = values.shape
        f = values.copy()
        for n in range(N):
            for d in range(D):
                last = 0.0
                for t in range(T):
                    if masks[n, t, d] > 0.5:
                        last = values[n, t, d]
                    else:
                        f[n, t, d] = last
                last = 0.0
                for t in range(T - 1, -1, -1):
                    if masks[n, t, d] > 0.5:
                        last = values[n, t, d]
                        break
                for t in range(T):
                    if masks[n, t, d] < 0.5 and f[n, t, d] == 0.0:
                        f[n, t, d] = last
        return f

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        x1 = self.targets[idx]
        nat_m = self.natural_masks[idx]
        rng = np.random.default_rng(self.seed + idx)
        obs_pos = np.where(nat_m > 0.5)
        n_obs = len(obs_pos[0])
        n_hide = int(n_obs * self.artificial_missing_rate)
        eval_m = np.zeros_like(nat_m)
        if n_hide > 0:
            hide = rng.choice(n_obs, size=n_hide, replace=False)
            eval_m[obs_pos[0][hide], obs_pos[1][hide]] = 1.0
        mask = nat_m * (1 - eval_m)
        x_obs = self.values[idx] * mask
        tg = compute_time_gaps(mask)
        return {
            "x1": torch.from_numpy(x1),
            "x_obs": torch.from_numpy(x_obs),
            "mask": torch.from_numpy(mask),
            "time_gaps": torch.from_numpy(tg),
            "eval_mask": torch.from_numpy(eval_m),
        }


def build_dataloaders(data, missing_rate=0.3, mechanism="mcar", batch_size=64,
                      train_ratio=0.8, seed=42, random_rate_train=True):
    n = len(data)
    n_train = int(n * train_ratio)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    train_data = data[perm[:n_train]]
    val_data = data[perm[n_train:]]

    if random_rate_train:
        train_ds = RandomRateDataset(train_data, seed=seed)
    else:
        train_ds = FixedMissDataset(train_data, missing_rate, mechanism, seed=seed)
    val_ds = FixedMissDataset(val_data, missing_rate, mechanism, seed=seed + 10000)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def build_physionet_dataloaders(values, natural_masks, artificial_missing_rate=0.1,
                                 batch_size=64, seed=42):
    n = len(values)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    tr = perm[:n_train]
    vl = perm[n_train:n_train + n_val]
    te = perm[n_train + n_val:]
    train_ds = PhysioNetDataset(values[tr], natural_masks[tr], artificial_missing_rate, seed=seed)
    val_ds = PhysioNetDataset(values[vl], natural_masks[vl], artificial_missing_rate, seed=seed + 10000)
    test_ds = PhysioNetDataset(values[te], natural_masks[te], artificial_missing_rate, seed=seed + 20000)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
    )
