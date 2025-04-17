# src/train.py
from __future__ import annotations
import torch, yaml, argparse, os
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from .data import split_loaders
from .schedule import DiffusionSchedule
from .ndp import NDP
from .metrics import rmse, pcrr


def evaluate(model_wrap, loader, device):
    y_true_all, y_pred_all = [], []
    model_wrap.model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_hat = model_wrap.sample(x)  # [B,N,1]
            y_true_all.append(y)
            y_pred_all.append(y_hat)
    y_true = torch.cat(y_true_all)
    y_pred = torch.cat(y_pred_all)
    return rmse(y_true, y_pred), pcrr(y_true, y_pred)


def build_scheduler(optimizer, cfg):
    """
    先 warm‑up 5% step 线性增 lr，再余弦退火到 0
    """
    total_steps = cfg["epochs"] * cfg["iter_per_epoch"]
    warmup_steps = int(0.05 * total_steps)
    warm = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=cfg.get("lr_min", 1e-6)
    )
    return SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup_steps])


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_dl, te_dl = split_loaders(cfg["csv"], cfg["batch"], split=0.8, seed=cfg["seed"])
    # 在 build_scheduler 里要用到每 epoch iteration 数
    cfg["iter_per_epoch"] = len(tr_dl)

    sched_diff = DiffusionSchedule(cfg["T"], device=device)
    ndp_wrap = NDP(
        cfg["D"],
        sched_diff,
        hidden=cfg["hidden"],
        n_layers=cfg["layers"],
        device=device,
    )
    model = ndp_wrap.model

    opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-2)
    lr_sched = build_scheduler(opt, cfg)

    best_rmse = float("inf")
    for epoch in range(cfg["epochs"]):
        model.train()
        pbar = tqdm(tr_dl, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            loss = ndp_wrap.loss(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_sched.step()  # <= 每 step 更新
            pbar.set_postfix(loss=loss.item(), lr=opt.param_groups[0]["lr"])

        # ------------ validation ------------
        rmse_val, pcrr_val = evaluate(ndp_wrap, te_dl, device)
        print(f"[Val] RMSE={rmse_val:.3f} | PCRR={pcrr_val:.3f}")

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/ndp_best.pt")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    train(cfg)
