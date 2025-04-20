#!/usr/bin/env python
# src/scripts/train_transfer.py
"""
迁移学习训练脚本 - 使用两个数据集作为训练集，一个数据集作为测试集
用于0样本测试场景
"""

from __future__ import annotations
import torch, yaml, argparse, os
from pathlib import Path
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import pandas as pd

# 导入项目模块
import sys

sys.path.append(str(Path(__file__).parents[2]))  # 添加项目根目录到路径
from src.data import RSMap
from src.ndp import NDP
from src.metrics import rmse, pcrr

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False


def load_multi_datasets(train_csv_paths, test_csv_path, batch_size):
    """
    加载多个训练集CSV文件和一个测试集CSV文件

    Args:
        train_csv_paths: 训练集CSV文件路径列表
        test_csv_path: 测试集CSV文件路径
        batch_size: 批次大小

    Returns:
        训练集DataLoader和测试集DataLoader
    """
    # 加载多个训练集并合并
    train_datasets = []
    for csv_path in train_csv_paths:
        ds = RSMap(Path(csv_path))
        train_datasets.append(ds)
        print(f"加载训练集: {csv_path}, 样本数: {len(ds)}")

    # 合并训练集
    combined_train_ds = ConcatDataset(train_datasets)
    print(f"合并后训练集样本总数: {len(combined_train_ds)}")

    # 加载测试集
    test_ds = RSMap(Path(test_csv_path))
    print(f"加载测试集: {test_csv_path}, 样本数: {len(test_ds)}")

    # 创建DataLoader
    train_dl = DataLoader(
        combined_train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dl, test_dl


def evaluate(model_wrap, loader, device, desc="[Eval]"):
    """评估模型性能"""
    y_true_all, y_pred_all = [], []
    model_wrap.model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device), y.to(device)
            y_hat = model_wrap.sample(x)
            y_true_all.append(y)
            y_pred_all.append(y_hat)
    y_true = torch.cat(y_true_all)
    y_pred = torch.cat(y_pred_all)
    return rmse(y_true, y_pred), pcrr(y_true, y_pred)


def build_scheduler(optimizer, cfg):
    """构建学习率调度器"""
    total_steps = cfg["epochs"] * cfg["iter_per_epoch"]
    warmup_steps = int(0.05 * total_steps)
    warm = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=cfg.get("lr_min", 1e-6)
    )
    return SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup_steps])


def train_transfer(cfg):
    """迁移学习训练函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载多个训练集和一个测试集
    train_csv_paths = cfg["train_csv_paths"]
    test_csv_path = cfg["test_csv_path"]
    tr_dl, te_dl = load_multi_datasets(train_csv_paths, test_csv_path, cfg["batch"])

    # 配置文件更新
    cfg["iter_per_epoch"] = len(tr_dl)

    # 创建保存目录
    save_dir = Path(cfg.get("save_dir", "results/transfer"))
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir / "runs")

    # 创建模型
    ndp_wrap = NDP(
        cfg["D"], cfg["T"], hidden=cfg["hidden"], n_layers=cfg["layers"], device=device
    )
    model = ndp_wrap.model

    # 如果需要从之前的检查点加载
    if "load_ckpt" in cfg and cfg["load_ckpt"]:
        print(f"从检查点加载模型: {cfg['load_ckpt']}")
        state_dict = torch.load(cfg["load_ckpt"], map_location=device)
        model.load_state_dict(state_dict)

    # 优化器和学习率调度
    opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-2)
    lr_sched = build_scheduler(opt, cfg)

    # 训练记录
    train_losses, val_rmses, val_pcrrs = [], [], []
    best_rmse = float("inf")

    # 训练循环
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

        # 计算平均损失
        avg_loss = total_loss / len(tr_dl)

        # 在测试集上评估
        rmse_val, pcrr_val = evaluate(ndp_wrap, te_dl, device, desc=f"[Test {epoch}]")

        # 记录结果
        train_losses.append(avg_loss)
        val_rmses.append(rmse_val)
        val_pcrrs.append(pcrr_val)

        # 记录到TensorBoard
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("RMSE/test", rmse_val, epoch)
        writer.add_scalar("PCRR/test", pcrr_val, epoch)

        # 打印结果
        print(f"[Train] Epoch {epoch} | Avg Loss={avg_loss:.4f}")
        print(f"[Test]  RMSE={rmse_val:.3f} | PCRR={pcrr_val:.3f}")

        # 保存最佳模型
        if rmse_val < best_rmse:
            best_rmse = rmse_val
            torch.save(model.state_dict(), ckpt_dir / "ndp_transfer_best.pt")
            print(f"✓ 保存最佳模型 (RMSE={best_rmse:.3f})")

    # 训练完成后绘制曲线
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, "b-", label="Train Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Training Loss")

    plt.subplot(2, 1, 2)
    plt.plot(val_rmses, "r-", label="Test RMSE")
    plt.plot(val_pcrrs, "g-", label="Test PCRR")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.title("Test Metrics")

    plt.tight_layout()
    plt.savefig(save_dir / "transfer_metrics_curve.png")
    plt.close()

    # 保存最终测试结果
    with open(save_dir / "transfer_results.txt", "w") as f:
        f.write(f"迁移学习结果 (训练: {train_csv_paths}, 测试: {test_csv_path})\n")
        f.write("=" * 60 + "\n")
        f.write(f"最佳测试 RMSE: {best_rmse:.6f}\n")
        f.write(f"最终测试 RMSE: {val_rmses[-1]:.6f}\n")
        f.write(f"最终测试 PCRR: {val_pcrrs[-1]:.6f}\n")
        f.write("=" * 60 + "\n")
        f.write("\n训练配置:\n")
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")

    print(f"✓ 训练完成，结果已保存到 {save_dir}")
    writer.close()

    return best_rmse


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="迁移学习训练脚本")
    parser.add_argument("--cfg", default="configs/transfer.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    train_transfer(cfg)
