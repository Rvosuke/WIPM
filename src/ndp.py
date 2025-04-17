from __future__ import annotations
import torch
from torch import nn
from .modules import TimeEmbedding, BiDimAttnBlock
from .schedule import DiffusionSchedule


class NDPNoisePredictor(nn.Module):
    """
    Implements ε_θ(x_t, y_t, t) exactly as Fig.2 本文
    输出尺寸 [B, N, 1]
    """

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
        # —— invariance layer：对D求和 → [B,N,H] → mlp→[B,N,1]
        self.post_norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(
        self, x: torch.Tensor, y_noisy: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        x       [B, N, D]
        y_noisy [B, N, 1]
        t       [B]    int
        """
        B, D = x.shape
        h_x = self.x_proj(x)  # [B,N,D_h]
        h_y = self.y_proj(y_noisy)  # [B,N,H]
        time = self.time_embed(t)  # [B,H]
        time = self.time_dense(time)  # [B,H]
        time = time.unsqueeze(1).unsqueeze(2)  # [B,1,1,H]
        h = h_x.unsqueeze(-1) + h_y.unsqueeze(2) + time  # [B,N,D,H]
        blk_res = 0
        for i, blk in enumerate(self.blocks):
            res = blk(h)
            blk_res = blk_res + res
            if i < len(self.blocks) - 1:
                h = res + h  # shape preserved
        h = blk_res
        # invariance over D：sum→[B,N,H]
        h = self.post_norm(h.sum(dim=2))
        return self.mlp(h)  # [B,N,1]


# ---------- wrapper: training / sampling ----------
class NDP:
    """
    简单封装：
      - .loss(batch)          -> scalar
      - .sample(x0, ctx=None) -> y0  (conditional / unconditional)
    """

    def __init__(self, in_dim, schedule: DiffusionSchedule, device, **model_kw):
        self.model = NDPNoisePredictor(in_dim, **model_kw)
        self.model.to(device)
        self.sched = schedule

    # ---- training step loss ----
    def loss(self, x0: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        device = x0.device
        B, _ = x0.shape
        t = torch.randint(0, self.sched.T, (B,), device=device)
        noise = torch.randn_like(y0)
        y_t = self.sched.q_sample(y0, t, noise)
        eps_pred = self.model(x0, y_t, t)
        diff = (eps_pred - noise).abs()
        return nn.functional.mse_loss(eps_pred, noise) + diff.mean()

    # ---- unconditional sample (prior) ----
    @torch.no_grad()
    def sample(self, x0: torch.Tensor) -> torch.Tensor:
        device = x0.device
        B, D = x0.shape
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
        return y_t  # y_0
