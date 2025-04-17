# src/ndp.py
from __future__ import annotations
import math, torch
from torch import nn, Tensor
from einops import rearrange
from dataclasses import dataclass


# ---------- Diffusion Schedule ----------
@dataclass
class DiffusionSchedule:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    device: str = "cpu"

    def __post_init__(self):
        self.T = self.timesteps
        self.betas = torch.linspace(
            self.beta_start, self.beta_end, self.timesteps, device=self.device
        )
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cum = torch.sqrt(1 - self.alpha_cumprod)

    def q_sample(self, y0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        a = self.sqrt_alpha_cum[t].unsqueeze(1)
        b = self.sqrt_one_minus_alpha_cum[t].unsqueeze(1)
        return a * y0 + b * noise

    def predict_prev(self, y_t: Tensor, eps_pred: Tensor, t: Tensor) -> Tensor:
        beta_t = self.betas[t].unsqueeze(1)
        alpha_t = self.alphas[t].unsqueeze(1)
        alpha_cum_t = self.alpha_cumprod[t].unsqueeze(1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cum[t].unsqueeze(1)

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / sqrt_one_minus
        return coef1 * (y_t - coef2 * eps_pred)


# ---------- sinusoidal time embedding ----------
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=device) * -math.log(10000.0) / half)
        emb = t.float().unsqueeze(1) * freqs
        return torch.cat((emb.sin(), emb.cos()), dim=1)


# ---------- BiDimAttnBlock (升级版) ----------
class BiDimAttnBlock(nn.Module):
    def __init__(self, hidden: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.mhsa_dim = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.mhsa_seq = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        B, N, D, H = s.shape

        s_d = rearrange(s, "b n d h -> (b n) d h")
        d_out, _ = self.mhsa_dim(self.norm1(s_d), self.norm1(s_d), self.norm1(s_d))
        d_out = rearrange(d_out, "(b n) d h -> b n d h", b=B, n=N)

        s_n = rearrange(s, "b n d h -> (b d) n h")
        n_out, _ = self.mhsa_seq(self.norm2(s_n), self.norm2(s_n), self.norm2(s_n))
        n_out = rearrange(n_out, "(b d) n h -> b n d h", b=B, d=D)

        out = s + d_out + n_out
        out = out + self.ffn(out)
        return out


class NDPNoisePredictor(nn.Module):
    def __init__(
        self, in_dim: int, hidden: int = 128, n_layers: int = 6, heads: int = 4
    ):
        super().__init__()
        self.hidden = hidden
        self.x_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.y_proj = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.time_embed = TimeEmbedding(hidden)
        self.time_dense = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList(
            [BiDimAttnBlock(hidden, heads) for _ in range(n_layers)]
        )
        self.post_norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        初始化是 Diffusion 模型稳定训练的关键。
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self, x: torch.Tensor, y_noisy: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        assert x.dim() == 2 and y_noisy.dim() == 2 and y_noisy.shape[1] == 1

        B, D = x.shape
        x = x.unsqueeze(1)  # [B, 1, D]
        y_noisy = y_noisy.unsqueeze(1)  # [B, 1, 1]

        h_x = self.x_proj(x)  # [B, 1, H]
        h_y = self.y_proj(y_noisy)  # [B, 1, H]
        time = self.time_embed(t)  # [B, H]
        time = self.time_dense(time).unsqueeze(1).unsqueeze(2)  # [B,1,1,H]
        h = h_x.unsqueeze(2) + h_y.unsqueeze(2) + time  # [B,1,1,H]

        for blk in self.blocks:
            h = blk(h)

        h = self.post_norm(h.sum(dim=2))  # [B, 1, H]
        return self.mlp(h).squeeze(2)  # [B, 1]


class NDP:
    def __init__(self, in_dim, time_step, device, **model_kw):
        self.model = NDPNoisePredictor(in_dim, **model_kw).to(device)
        self.sched = DiffusionSchedule(timesteps=time_step, device=device)

    def loss(self, x0: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        assert x0.dim() == 2 and y0.shape[-1] == 1
        device = x0.device
        B = x0.shape[0]
        t = torch.randint(0, self.sched.T, (B,), device=device)
        noise = torch.randn_like(y0)
        y_t = self.sched.q_sample(y0, t, noise)
        eps_pred = self.model(x0, y_t, t)
        return nn.functional.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, x0: torch.Tensor) -> torch.Tensor:
        device = x0.device
        B = x0.shape[0]
        y_t = torch.randn(B, 1, device=device)
        for step in reversed(range(self.sched.T)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            eps = self.model(x0, y_t, t)
            y_prev = self.sched.predict_prev(y_t, eps, t)
            if step > 0:
                noise = torch.randn_like(y_t)
                beta = self.sched.betas[step].sqrt().to(device)
                y_prev += beta * noise
            y_t = y_prev
        return y_t
