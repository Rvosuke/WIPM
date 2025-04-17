from __future__ import annotations
import torch
from typing import Dict


class DiffusionSchedule:
    """
    Implements β_t table and helper functions
    matching DDPM / NDP equations.
    """

    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device="cpu",
    ) -> None:
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)  # [T]
        self.alphas = 1.0 - self.betas  # [T]
        self.alpha_cum = torch.cumprod(self.alphas, dim=0)  # ᾱ_t
        self.sqrt_alpha_cum = torch.sqrt(self.alpha_cum)
        self.sqrt_one_minus = torch.sqrt(1 - self.alpha_cum)

    # ---------- forward (q) ----------
    def q_sample(
        self, y0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        y_t = √ᾱ_t y0 + √(1-ᾱ_t) ε
        """
        a = self.sqrt_alpha_cum[t].unsqueeze(-1).unsqueeze(-1)  # [...,1,1]
        b = self.sqrt_one_minus[t].unsqueeze(-1).unsqueeze(-1)
        return a * y0 + b * noise

    # ---------- reverse ----------
    def predict_prev(
        self, y_t: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        beta = self.betas[t].unsqueeze(-1).unsqueeze(-1)
        alpha = self.alphas[t].unsqueeze(-1).unsqueeze(-1)
        alpha_c = self.alpha_cum[t].unsqueeze(-1).unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus[t].unsqueeze(-1).unsqueeze(-1)
        coef1 = 1.0 / torch.sqrt(alpha)
        coef2 = beta / sqrt_one_minus
        return coef1 * (y_t - coef2 * eps)

    # make object cuda‑friendly
    def to(self, device):
        return DiffusionSchedule(
            self.T, self.betas[0].item(), self.betas[-1].item(), device=device
        )
