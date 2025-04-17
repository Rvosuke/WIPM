# src/metrics.py
import torch
from .scripts.preprocess import rsrp_norm

P_TH = -103


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torch.sqrt(((y_true - y_pred) ** 2).mean()).item()


def pcrr(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    t = rsrp_norm(P_TH)
    tp = ((y_true < t) & (y_pred < t)).sum()
    fp = ((y_true >= t) & (y_pred < t)).sum()
    fn = ((y_true < t) & (y_pred >= t)).sum()

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return (2 * prec * rec / (prec + rec + 1e-9)).item()
