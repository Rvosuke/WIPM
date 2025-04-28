# src/train.py
import torch, yaml, argparse
import pandas as pd
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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
            y_hat = model_wrap.sample(x, seq_len=y.shape[1])
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


def visualize(cfg, train_losses, val_rmses, val_pcrrs, save_dir):
    # 绘图保存历史记录
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 绘制训练损失
    ax1.plot(range(cfg["epochs"]), train_losses, "b-", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss Curve")
    ax1.grid(True)
    ax1.legend()

    # 绘制验证指标
    val_epochs = list(
        range(10, cfg["epochs"] + 1, 10)
    )  # 验证指标是在第9、19、29...个epoch记录的
    ax2.plot(val_epochs, val_rmses, "r-", marker="o", label="Validation RMSE")
    ax2.plot(val_epochs, val_pcrrs, "g-", marker="s", label="Validation PCRR")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric Value")
    ax2.set_title("Validation Metrics Curve")
    ax2.grid(True)
    ax2.legend()

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_dir / "metrics_curve.png", dpi=300)
    plt.close()


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_dl, te_dl = split_loaders(
        cfg["csv"], cfg["batch"], split=cfg["train_rate"], seed=cfg["seed"]
    )
    cfg["iter_per_epoch"] = len(tr_dl)

    save_dir = Path(cfg.get("save_dir", "results"))
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
            loss = ndp_wrap.loss(x.to(device), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_sched.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), lr=opt.param_groups[0]["lr"])

        avg_loss = total_loss / len(tr_dl)
        train_losses.append(avg_loss)
        print(f"[Train] Epoch {epoch} | Avg Loss={avg_loss:.4f}")

        if epoch % 10 == 9:
            rmse_val, pcrr_val = evaluate(ndp_wrap, te_dl, device)
            val_rmses.append(rmse_val)
            val_pcrrs.append(pcrr_val)
            print(f"[Val]   RMSE={rmse_val:.3f} | PCRR={pcrr_val:.3f}")

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            torch.save(model.state_dict(), ckpt_dir / "ndp_best.pt")
    visualize(cfg, train_losses, val_rmses, val_pcrrs, save_dir)
    val_metrics_df = pd.DataFrame(
        {
            "Epoch": range(10, cfg["epochs"] + 1, 10),
            "RMSE": val_rmses,
            "PCRR": val_pcrrs,
        }
    )
    val_metrics_df.to_csv(save_dir / "validation_metrics.csv", index=False)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    train(cfg)
