# src/ndp.py
from __future__ import annotations
import math, torch
from torch import nn, Tensor
from einops import rearrange
from dataclasses import dataclass


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
        """
        用于在训练时对数据进行加噪声处理
        Args:
            y0: 输入特征，形状为[batch, out_len, out_dim]
            t: 扩散时间步，形状为[batch]
            noise: 噪声，形状为[batch, out_len, out_dim]
        Returns:
            加噪声后的目标值，形状为[batch, out_len, out_dim]
        """
        a = self.sqrt_alpha_cum[t].unsqueeze(-1).unsqueeze(-1)
        b = self.sqrt_one_minus_alpha_cum[t].unsqueeze(-1).unsqueeze(-1)
        return a * y0 + b * noise

    def predict_prev(self, y_t: Tensor, eps_pred: Tensor, t: Tensor) -> Tensor:
        beta_t = self.betas[t].unsqueeze(-1).unsqueeze(-1)
        alpha_t = self.alphas[t].unsqueeze(-1).unsqueeze(-1)
        # alpha_cum_t = self.alpha_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alpha_cum[t].unsqueeze(-1).unsqueeze(-1)

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / sqrt_one_minus
        return coef1 * (y_t - coef2 * eps_pred)


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


class BiDimAttnBlock(nn.Module):
    def __init__(self, hidden: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.mhsa_dim = nn.MultiheadAttention(
            hidden, num_heads, batch_first=True, dropout=dropout
        )
        self.mhsa_seq = nn.MultiheadAttention(
            hidden, num_heads, batch_first=True, dropout=dropout
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        B, N, D, H = s.shape

        s_d = rearrange(s, "b n d h -> (b n) d h")
        d_out, _ = self.mhsa_dim(self.norm1(s_d), self.norm1(s_d), self.norm1(s_d))
        d_out = self.dropout1(d_out)
        d_out = rearrange(d_out, "(b n) d h -> b n d h", b=B, n=N)

        s_n = rearrange(s, "b n d h -> (b d) n h")
        n_out, _ = self.mhsa_seq(self.norm2(s_n), self.norm2(s_n), self.norm2(s_n))
        n_out = self.dropout2(n_out)
        n_out = rearrange(n_out, "(b d) n h -> b n d h", b=B, d=D)

        out = s + d_out + n_out
        out = out + self.dropout3(self.ffn(out))
        return out


class NDPNoisePredictor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden: int = 256,
        n_layers: int = 6,
        heads: int = 4,
    ):
        super().__init__()
        self.hidden = hidden
        self.x_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.y_proj = nn.Sequential(
            nn.Linear(out_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.time_embed = TimeEmbedding(hidden)
        self.time_dense = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.blocks = nn.ModuleList(
            [BiDimAttnBlock(hidden, heads) for _ in range(n_layers)]
        )
        self.post_norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
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
        """
        Args:
            x: 输入特征，形状为[batch, in_len, in_dim]
            y_noisy: 带噪声的目标值，形状为[batch, out_len, out_dim]
            t: 扩散时间步，形状为[batch]

        Returns:
            预测值, 与y_noisy形状相同
        """
        h_x = self.x_proj(x)  # [batch, in_len, in_dim] -> [batch, in_len, dim_hid]
        h_y = self.y_proj(y_noisy)  # [..., out_dim] -> [batch, in_len, dim_hid]
        time = self.time_embed(t)  # [batch] -> [batch, dim_hid]
        time = self.time_dense(time).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, dim_hid]
        hidden = h_x.unsqueeze(2) + h_y.unsqueeze(1) + time
        del h_x, h_y, time

        for blk in self.blocks:
            hidden = blk(hidden)

        hidden = self.post_norm(hidden.sum(dim=1))  # [batch, out_len, dim_hid]
        return self.mlp(hidden)


class NDP:
    def __init__(self, in_dim, time_step, device, **model_kw):
        self.model = NDPNoisePredictor(in_dim, **model_kw).to(device)
        self.sched = DiffusionSchedule(timesteps=time_step, device=device)

    def loss(self, x0: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        """
         计算时间序列扩散模型的损失函数

        Args:
            x0: 输入特征，形状为[B, N_in, D]
            y0: 目标值，形状为[B, N_out, 1]

        Returns:
            mse损失
        """
        device = x0.device
        B = x0.shape[0]

        # 单步预测和序列预测统一处理
        t = torch.randint(0, self.sched.T, (B,), device=device)
        noise = torch.randn_like(y0)
        y_t = self.sched.q_sample(y0, t, noise)
        eps_pred = self.model(x0, y_t, t)
        return nn.functional.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, x0: torch.Tensor, seq_len: int, out_dim: int = 1) -> torch.Tensor:
        """
        多步去噪以完整地预测整个时间序列, 常用于推理。

        Args:
            x0: 输入特征，形状为[B, N_in, D]
            seq_len: 要预测的序列长度

        Returns:
            预测序列，形状为[B, seq_len, 1]
        """
        device = x0.device
        B = x0.shape[0]

        # 初始化带噪声的输出序列
        y_t = torch.randn(B, seq_len, out_dim, device=device)

        # 逐步去噪过程
        for step in reversed(range(self.sched.T)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            eps = self.model(x0, y_t, t)
            y_prev = self.sched.predict_prev(y_t, eps, t)
            if step > 0:
                noise = torch.randn_like(y_t, device=device)
                beta = self.sched.betas[step].sqrt().to(device)
                y_prev += beta * noise
            y_t = y_prev
        return y_t
