#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5G Traffic 时间序列预测实验脚本
用于执行基于NDP模型的5G流量数据预测实验
"""

import sys, argparse, torch, yaml, warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.ndp import NDP
from src.data import TrafficSeries
from src.utils import build_scheduler


@torch.no_grad()
def evaluate(model, test_loader, device):
    """评估模型性能"""
    model.model.eval()
    all_preds, all_targets = [], []
    for x, y in test_loader:
        y_pred = model.sample(x.to(device), y.size(1))
        all_preds.append(y_pred)
        all_targets.append(y.to(device))
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    mse = torch.mean((all_preds - all_targets) ** 2).item()
    mae = torch.mean(torch.abs(all_preds - all_targets)).item()
    print(f"[Val] MSE: {mse:.4f}, MAE: {mae:.4f}")
    return mse, mae, all_preds.cpu().numpy(), all_targets.cpu().numpy()


def visualize_sample(preds, targets, save_path, sample_idx=0):
    """可视化预测结果"""
    pred_sample = preds[
        sample_idx
    ].flatten()  # 将形状从 [pred_window, 1] 转换为 [pred_window]
    target_sample = targets[sample_idx].flatten()
    x_points = np.arange(len(pred_sample))  # 创建X轴坐标

    plt.style.use("ggplot")
    plt.grid(True, linestyle="-", alpha=0.7)
    plt.figure(figsize=(12, 7))

    plt.plot(x_points, target_sample, "b-", label="Ground Truth")
    plt.plot(x_points, pred_sample, "r--", label="Prediction")
    # 添加标题和标签
    plt.title(f"Sample Prediction")
    plt.xlabel("X")
    plt.ylabel("Value")
    # 添加图例
    plt.legend(fontsize=12, loc="upper right")
    # 设置Y轴范围
    ymin = min(min(pred_sample), min(target_sample)) - 0.1
    ymax = max(max(pred_sample), max(target_sample)) + 0.1
    plt.ylim([ymin, ymax])

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"样本 {sample_idx} 的评估指标:")
    print(f"MSE: {np.mean((pred_sample - target_sample)**2):.4f}")
    print(f"MAE: {np.mean(np.abs(pred_sample - target_sample)):.4f}")
    print(f"RMSE: {np.sqrt(np.mean((pred_sample - target_sample)**2)):.4f}")
    print(f"📊 预测可视化已保存至: {save_path}")


def noise_robustness_test(
    model,
    test_dataset,
    device,
    noise_levels=[0.01, 0.02, 0.03, 0.04],
    batch_size=64,
    save_dir=None,
):
    """噪声鲁棒性测试"""
    results = []

    for noise_level in noise_levels:
        print(f"\n🧪 噪声水平 {noise_level} 的鲁棒性测试")

        # 添加噪声到测试数据
        test_dataset.data += np.random.normal(0, noise_level, test_dataset.data)
        test_dataset._prepare_sequences()

        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        # 评估在噪声数据上的性能
        mse, mae, _, _ = evaluate(model, test_loader, device)

        # 记录结果
        results.append({"noise_level": noise_level, "mse": mse, "mae": mae})

        # # 可视化(如果提供了保存目录)
        # if save_dir:
        #     plot_path = save_dir / f"noise_robustness_{noise_level}.png"
        #     visualize_predictions(
        #         preds,
        #         targets,
        #         test_dates,
        #         plot_path,
        #         f"噪声水平 {noise_level} 的预测结果",
        #     )

    # 绘制噪声-性能曲线
    if save_dir:
        plt.figure(figsize=(12, 10))
        noise_levels = [r["noise_level"] for r in results]
        mse_values = [r["mse"] for r in results]
        mae_values = [r["mae"] for r in results]

        # 创建两个子图
        plt.subplot(2, 1, 1)
        plt.plot(noise_levels, mse_values, "o-", color="blue")
        plt.title("MSE vs Noise Level")
        plt.xlabel("Noise Level (standard deviation)")
        plt.ylabel("MSE")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(noise_levels, mae_values, "s-", color="red")
        plt.title("MAE vs Noise Level")
        plt.xlabel("Noise Level (sigma)")
        plt.ylabel("MAE")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_dir / "noise_performance_curve.png")

        # 保存结果到文本文件
        with open(save_dir / "noise_robustness_results.txt", "w") as f:
            f.write("噪声水平测试结果:\n")
            for r in results:
                f.write(
                    f"噪声水平: {r['noise_level']}, MSE: {r['mse']:.4f}, MAE: {r['mae']:.4f}\n"
                )

    return results


def main(cfg):
    device, epochs = str(cfg["device"]), cfg["epochs"]
    save_dir = (
        Path(cfg["save_dir"])
        / Path(cfg["csv"]).stem
        / f"out{cfg["output_len"]}_rate{cfg["train_rate"]}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpts_dirs = save_dir / "checkpoints"
    ckpts_dirs.mkdir(exist_ok=True)

    tr_ds = TrafficSeries(
        cfg["csv"], cfg["input_len"], cfg["output_len"], cfg["train_rate"], train=True
    )
    te_ds = TrafficSeries(
        cfg["csv"], cfg["input_len"], cfg["output_len"], cfg["train_rate"], train=False
    )
    tr_dl = DataLoader(tr_ds, batch_size=cfg["batch"], shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=cfg["batch"], shuffle=False)

    cfg["iter_per_epoch"] = len(tr_dl)

    ndp_wrap = NDP(
        cfg["D"],
        cfg["T"],
        device,
        hidden=cfg["hidden"],
        n_layers=cfg["layers"],
        in_len=cfg["input_len"],
        out_len=cfg["output_len"],
    )
    model = ndp_wrap.model
    if cfg["is_train"]:
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        lr_sched = build_scheduler(opt, cfg)

        train_losses, val_mses, val_maes = [], [], []
        best_mse = float("inf")
        for epoch in range(epochs):
            avg_loss = ndp_wrap.train_epoch(tr_dl, epoch, opt, device, lr_sched)
            train_losses.append(avg_loss)
            print(f"[Train] Epoch {epoch} | Avg Loss={avg_loss:.4f}")

            mse, mae, _, _ = evaluate(ndp_wrap, te_dl, device)
            val_mses.append(mse)
            val_maes.append(mae)
            if mse < best_mse:
                best_mse = mse
                torch.save(model.state_dict(), ckpts_dirs / "ndp_best.pt")
        # 绘制训练历史曲线
        plt.figure(figsize=(10, 6))
        epochs_range = range(1, epochs + 1)
        plt.plot(epochs_range, train_losses, label="Train Loss", color="blue")
        plt.plot(epochs_range, val_mses, label="Val MSE", color="red")
        plt.plot(epochs_range, val_maes, label="Val MAE", color="green")
        plt.title("Training and Validation Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / "metrics_curve.png")
    # 最终评估, 在最佳模型
    model.load_state_dict(torch.load(ckpts_dirs / "ndp_best.pt", map_location=device))
    mse, mae, preds, targets = evaluate(ndp_wrap, te_dl, device)

    # 可视化预测样本
    visualize_sample(preds, targets, save_dir / "final_prediction.png", 10)

    if cfg["noise_test"]:
        # 进行噪声鲁棒性测试
        noise_robustness_test(
            ndp_wrap,
            te_ds,
            cfg["input_len"],
            cfg["output_len"],
            device,
            save_dir=save_dir,
        )

    # 保存最终结果
    final_results = {
        "final_mse": mse,
        "final_mae": mae,
    }

    with open(save_dir / "final_results.txt", "w") as f:
        f.write("最终实验结果:\n")
        for k, v in final_results.items():
            f.write(f"{k}: {v}\n")

    print(f"\n✅ 实验完成! 结果已保存到 {save_dir}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="5G流量预测实验")
    parser.add_argument("--cfg", default="configs/traffic.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    main(cfg)
