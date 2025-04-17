"""
封装 forward / reverse 过程与调度。
只实现简单 1D 标签扩散；x 不加噪 (conditional NDP)。
"""

from __future__ import annotations
import torch
from torch import Tensor
from dataclasses import dataclass
import math


@dataclass
class DiffusionSchedule:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def __post_init__(self):
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cum = torch.sqrt(1 - self.alpha_cumprod)

    def q_sample(self, y0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """
        公式: y_t = √ᾱ_t y0 + √(1-ᾱ_t) ε
        """
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
