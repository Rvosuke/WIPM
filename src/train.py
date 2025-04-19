# src/train.py
from __future__ import annotations
import torch, yaml, argparse, os
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from .data import split_loaders
from .ndp import NDP
from .metrics import rmse, pcrr


def evaluate(model_wrap, loader, device):
    y_true_all, y_pred_all = [], []
    model_wrap.model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="[Eval]", leave=False):
            x, y = x.to(device), y.to(device)
            y_hat = model_wrap.sample(x)
            y_true_all.append(y)
            y_pred_all.append(y_hat)
    y_true = torch.cat(y_true_all)
    y_pred = torch.cat(y_pred_all)
    return rmse(y_true, y_pred), pcrr(y_true, y_pred)


def build_scheduler(optimizer, cfg):
    total_steps = cfg["epochs"] * cfg["iter_per_epoch"]
    warmup_steps = int(0.05 * total_steps)
    warm = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=cfg.get("lr_min", 1e-6)
    )
    return SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup_steps])


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_dl, te_dl = split_loaders(cfg["csv"], cfg["batch"], split=0.9, seed=cfg["seed"])
    cfg["iter_per_epoch"] = len(tr_dl)

    save_dir = Path(cfg.get("save_dir", "results"))
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir / "runs")

    ndp_wrap = NDP(
        cfg["D"], cfg["T"], hidden=cfg["hidden"], n_layers=cfg["layers"], device=device
    )
    model = ndp_wrap.model
    opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-2)
    lr_sched = build_scheduler(opt, cfg)

    train_losses, val_rmses, val_pcrrs = [], [], []
    best_rmse = float("inf")

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0
        pbar = tqdm(tr_dl, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            loss = ndp_wrap.loss(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_sched.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), lr=opt.param_groups[0]["lr"])

        avg_loss = total_loss / len(tr_dl)
        rmse_val, pcrr_val = evaluate(ndp_wrap, te_dl, device)

        train_losses.append(avg_loss)
        val_rmses.append(rmse_val)
        val_pcrrs.append(pcrr_val)

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("RMSE/val", rmse_val, epoch)
        writer.add_scalar("PCRR/val", pcrr_val, epoch)

        print(f"[Train] Epoch {epoch} | Avg Loss={avg_loss:.4f}")
        print(f"[Val]   RMSE={rmse_val:.3f} | PCRR={pcrr_val:.3f}")

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            torch.save(model.state_dict(), ckpt_dir / "ndp_best.pt")

    # 可选：绘图保存
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_rmses, label="Val RMSE")
    plt.plot(val_pcrrs, label="Val PCRR")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.title("Training Progress")
    plt.savefig(save_dir / "metrics_curve.png")
    plt.close()

    writer.close()


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    train(cfg)
