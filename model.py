"""MissPath-FM: missingness-aware probability paths for flow matching.

Components
----------
1. Source priors            : InterpolationPrior (default), GaussianPrior (ablation)
2. Probability paths        : MissingnessAwarePath (default), StandardLinearPath (ablation)
3. Velocity model           : Transformer over (x_t, t, mask, time_gaps)
4. MissPathFM               : the full model (training loss + ODE sampling)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Source priors
# ---------------------------------------------------------------------------

class GaussianPrior(nn.Module):
    """Standard flow-matching source: z_0 ~ N(0, I). Used for the prior ablation."""

    def sample(self, x_obs, mask, time_gaps=None):
        return torch.randn_like(x_obs)


class InterpolationPrior(nn.Module):
    """Observation-aware source prior.

    z_0 ~ N(mu(x_obs, mask), sigma(mask))
      mu    : linear interpolation between neighbouring observed values
      sigma : sigma_obs at observed positions, sigma_miss at missing positions

    The interpolation is a vectorised forward-fill / backward-fill pass, so the
    cost is O(T) tensor ops rather than a per-element Python loop.
    """

    def __init__(self, sigma_obs=0.1, sigma_miss=1.0):
        super().__init__()
        self.sigma_obs = sigma_obs
        self.sigma_miss = sigma_miss

    def _interpolate(self, x_obs, mask):
        B, T, D = x_obs.shape
        device = x_obs.device

        # Forward pass: last observed value / its time index / whether one exists
        fwd_val = torch.zeros(B, 1, D, device=device)
        fwd_time = torch.zeros(B, 1, D, device=device)
        fwd_valid = torch.zeros(B, 1, D, device=device)
        fwd_vals, fwd_times, fwd_valids = [], [], []

        for t in range(T):
            m_t = mask[:, t:t + 1, :]
            x_t = x_obs[:, t:t + 1, :]
            t_tensor = torch.full_like(m_t, t)
            fwd_val = torch.where(m_t > 0.5, x_t, fwd_val)
            fwd_time = torch.where(m_t > 0.5, t_tensor, fwd_time)
            fwd_valid = torch.where(m_t > 0.5, torch.ones_like(fwd_valid), fwd_valid)
            fwd_vals.append(fwd_val.clone())
            fwd_times.append(fwd_time.clone())
            fwd_valids.append(fwd_valid.clone())

        fwd_vals = torch.cat(fwd_vals, dim=1)
        fwd_times = torch.cat(fwd_times, dim=1)
        fwd_valids = torch.cat(fwd_valids, dim=1)

        # Backward pass: next observed value / its time index / existence
        bwd_val = torch.zeros(B, 1, D, device=device)
        bwd_time = torch.full((B, 1, D), T - 1, dtype=torch.float32, device=device)
        bwd_valid = torch.zeros(B, 1, D, device=device)
        bwd_vals, bwd_times, bwd_valids = [], [], []

        for t in range(T - 1, -1, -1):
            m_t = mask[:, t:t + 1, :]
            x_t = x_obs[:, t:t + 1, :]
            t_tensor = torch.full_like(m_t, t)
            bwd_val = torch.where(m_t > 0.5, x_t, bwd_val)
            bwd_time = torch.where(m_t > 0.5, t_tensor, bwd_time)
            bwd_valid = torch.where(m_t > 0.5, torch.ones_like(bwd_valid), bwd_valid)
            bwd_vals.append(bwd_val.clone())
            bwd_times.append(bwd_time.clone())
            bwd_valids.append(bwd_valid.clone())

        bwd_vals = torch.cat(bwd_vals[::-1], dim=1)
        bwd_times = torch.cat(bwd_times[::-1], dim=1)
        bwd_valids = torch.cat(bwd_valids[::-1], dim=1)

        t_grid = torch.arange(T, device=device).float().view(1, T, 1).expand(B, T, D)
        dt = (bwd_times - fwd_times).clamp(min=1)
        alpha = (t_grid - fwd_times) / dt

        both_valid = fwd_valids * bwd_valids
        mu = torch.where(
            both_valid > 0.5,
            (1 - alpha) * fwd_vals + alpha * bwd_vals,          # interpolate
            torch.where(fwd_valids > 0.5, fwd_vals,             # extrapolate right
                        torch.where(bwd_valids > 0.5, bwd_vals,  # extrapolate left
                                    torch.zeros_like(x_obs))),   # channel never observed
        )
        # Exact values at observed positions
        mu = torch.where(mask > 0.5, x_obs, mu)
        return mu

    def sample(self, x_obs, mask, time_gaps=None):
        mu = self._interpolate(x_obs, mask)
        sigma = mask * self.sigma_obs + (1 - mask) * self.sigma_miss
        return mu + sigma * torch.randn_like(mu)


# ---------------------------------------------------------------------------
# Probability paths
# ---------------------------------------------------------------------------

class StandardLinearPath(nn.Module):
    """x_t = (1 - t) z_0 + t x_1, u_t = x_1 - z_0. Used for the path ablation."""

    def sample_path(self, z_0, x_1, t, mask=None):
        x_t = (1 - t) * z_0 + t * x_1
        return x_t, x_1 - z_0


class MissingnessAwarePath(nn.Module):
    """Speed-modulated path: alpha(t) = 1 - (1 - t) ** s(c).

    s(c) = speed_obs at observed positions, speed_miss at missing positions.
    Observed positions converge faster; missing positions keep the standard
    linear schedule and therefore retain more transport time.

        x_t = (1 - alpha(t)) z_0 + alpha(t) x_1
        u_t = s(c) * (1 - t) ** (s(c) - 1) * (x_1 - z_0)
    """

    def __init__(self, speed_obs=2.0, speed_miss=1.0):
        super().__init__()
        self.speed_obs = speed_obs
        self.speed_miss = speed_miss

    def sample_path(self, z_0, x_1, t, mask=None):
        if mask is None:
            return (1 - t) * z_0 + t * x_1, x_1 - z_0
        speed = mask * self.speed_obs + (1 - mask) * self.speed_miss
        one_minus_t = (1 - t).clamp(min=1e-6)
        alpha_t = 1 - one_minus_t ** speed
        x_t = (1 - alpha_t) * z_0 + alpha_t * x_1
        target_v = speed * (one_minus_t ** (speed - 1)) * (x_1 - z_0)
        return x_t, target_v


# ---------------------------------------------------------------------------
# Velocity model
# ---------------------------------------------------------------------------

class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of the flow time t, followed by an MLP."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        t = t.view(-1)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class VelocityModel(nn.Module):
    """Transformer velocity field v_theta(x_t, t, c).

    context controls which parts of the observation pattern c are fed in:
      "full"      : x_t + mask + time gaps (default)
      "mask_only" : x_t + mask
      "gap_only"  : x_t + time gaps
      "no_mask"   : x_t only
    """

    CONTEXT_MULTIPLIER = {"no_mask": 1, "mask_only": 2, "gap_only": 2, "full": 3}

    def __init__(self, input_dim, hidden_dim=256, n_heads=4, n_layers=4,
                 dropout=0.1, context="full"):
        super().__init__()
        if context not in self.CONTEXT_MULTIPLIER:
            raise ValueError(f"Unknown context mode: {context}")
        self.context = context

        self.input_proj = nn.Linear(input_dim * self.CONTEXT_MULTIPLIER[context], hidden_dim)
        self.time_embed = TimeEmbedding(hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x_t, t, mask, time_gaps):
        B, T, D = x_t.shape
        tg_norm = time_gaps / (time_gaps.max() + 1)
        if self.context == "full":
            inp = torch.cat([x_t, mask, tg_norm], dim=-1)
        elif self.context == "mask_only":
            inp = torch.cat([x_t, mask], dim=-1)
        elif self.context == "gap_only":
            inp = torch.cat([x_t, tg_norm], dim=-1)
        else:
            inp = x_t

        h = self.input_proj(inp)
        h = h + self.time_embed(t).unsqueeze(1)
        h = h + self.pos_embed[:, :T, :]
        h = self.transformer(h)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MissPathFM(nn.Module):
    """MissPath-FM: observation-aware prior + missingness-aware path + velocity field."""

    def __init__(self, input_dim, hidden_dim=256, n_heads=4, n_layers=4,
                 prior="interpolation", path="missingness_aware", context="full",
                 sigma_obs=0.1, sigma_miss=1.0, speed_obs=2.0, speed_miss=1.0,
                 obs_consistency_weight=0.1, dropout=0.1):
        super().__init__()

        if prior == "interpolation":
            self.prior = InterpolationPrior(sigma_obs=sigma_obs, sigma_miss=sigma_miss)
        elif prior == "gaussian":
            self.prior = GaussianPrior()
        else:
            raise ValueError(f"Unknown prior: {prior}")

        if path == "missingness_aware":
            self.path = MissingnessAwarePath(speed_obs=speed_obs, speed_miss=speed_miss)
        elif path == "standard":
            self.path = StandardLinearPath()
        else:
            raise ValueError(f"Unknown path: {path}")

        self.velocity = VelocityModel(input_dim, hidden_dim, n_heads, n_layers,
                                      dropout=dropout, context=context)
        self.prior_name = prior
        self.path_name = path
        self.context = context
        self.obs_consistency_weight = obs_consistency_weight

    def compute_loss(self, batch):
        """Conditional flow-matching loss + observation-consistency penalty."""
        x_1 = batch["x1"]
        x_obs = batch["x_obs"]
        mask = batch["mask"]
        time_gaps = batch["time_gaps"]
        B = x_1.shape[0]

        z_0 = self.prior.sample(x_obs, mask, time_gaps)
        t = torch.rand(B, device=x_1.device)
        t_exp = t.view(B, 1, 1)
        x_t, target_v = self.path.sample_path(z_0, x_1, t_exp, mask)
        pred_v = self.velocity(x_t, t, mask, time_gaps)

        cfm_loss = F.mse_loss(pred_v, target_v)
        obs_loss = (mask * (pred_v - target_v) ** 2).sum() / (mask.sum() + 1e-8)
        loss = cfm_loss + self.obs_consistency_weight * obs_loss
        return {"loss": loss, "cfm_loss": cfm_loss.item(), "obs_loss": obs_loss.item()}

    @torch.no_grad()
    def sample(self, x_obs, mask, time_gaps, n_steps=20):
        """Integrate the learned velocity field with an Euler solver."""
        x = self.prior.sample(x_obs, mask, time_gaps)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i / n_steps, device=x.device)
            v = self.velocity(x, t, mask, time_gaps)
            x = x + v * dt
        return mask * x_obs + (1 - mask) * x
