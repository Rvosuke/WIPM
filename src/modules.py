from __future__ import annotations
import torch, math
from torch import nn
from einops import rearrange


# ---------- sinusoidal time embedding ----------
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B]  int64
        Return:
            [B, dim]
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device) * -math.log(10000.0) / half
        )  # [half,]
        emb = t.float().unsqueeze(1) * freqs  # [B,half]
        return torch.cat((emb.sin(), emb.cos()), dim=1)  # [B,dim]


# ---------- bi‑dimensional attention block ----------
class BiDimAttnBlock(nn.Module):
    """
    Two MHSA layers working on:
      - dimension axis D
      - sequence axis N
    Follows NDP Sec.4
    """

    def __init__(self, hidden: int, num_heads: int = 4) -> None:
        super().__init__()
        self.mhsa_dim = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.mhsa_seq = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.act = nn.ReLU()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: [B, N, D, H]
        Returns same shape
        """
        B, N, D, H = s.shape
        # ----- attention over D -----
        s_d = rearrange(s, "b n d h -> (b n) d h")  # treat N并批
        d_out, _ = self.mhsa_dim(s_d, s_d, s_d)  # [B*N, D, H]
        d_out = rearrange(d_out, "(b n) d h -> b n d h", b=B, n=N)

        # ----- attention over N -----
        s_n = rearrange(s, "b n d h -> (b d) n h")  # treat D并批
        n_out, _ = self.mhsa_seq(s_n, s_n, s_n)  # [B*D, N, H]
        n_out = rearrange(n_out, "(b d) n h -> b n d h", b=B, d=D)

        out = s + self.act(self.norm(d_out + n_out))
        return out
