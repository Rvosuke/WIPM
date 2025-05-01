#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5G Traffic 时间序列预测实验脚本
用于执行基于NDP模型的5G流量数据预测实验
"""

import os, sys, argparse, torch, yaml, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.ndp import NDP
from src.train import train_epoch, build_scheduler


class TimeSeriesDataset(Dataset):
    """时间序列数据集类 处理5G流量数据"""

    def __init__(self, data, input_window=12, pred_window=24, stride=1):
        """
        初始化时间序列数据集

        Args:
            data: 原始时间序列数据（已归一化）
            input_window: 输入窗口大小(默认72小时-3天)
            pred_window: 预测窗口大小(长期预测72小时,短期预测36小时)
            stride: 滑动窗口步长(默认为1)
        """
        self.data = data
        self.input_window = input_window
        self.pred_window = pred_window
        self.stride = stride
        self._prepare_sequences()

    def _prepare_sequences(self):
        """准备输入序列和目标序列"""
        self.samples = []
        total_len = len(self.data)

        for i in range(
            0, total_len - self.input_window - self.pred_window + 1, self.stride
        ):
            x_seq = self.data[i : i + self.input_window]
            y_seq = self.data[
                i + self.input_window : i + self.input_window + self.pred_window, 0:1
            ]  # 只取value列作为预测目标
            self.samples.append((x_seq, y_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        # 返回形状为[input_window, feature_dim]的输入和形状为[pred_window, 1]的输出
        return torch.FloatTensor(x), torch.FloatTensor(y)


def load_and_preprocess(file_path, feature_cols=["value", "ma_value", "lg_ma_value"]):
    """
    加载并预处理5G流量数据, 首先检查数据完整性, 提取特征并进行标准化

    Args:
        file_path: CSV文件路径
        feature_cols: 使用的特征列

    Returns:
        预处理后的数据，特征缩放器
    """
    print(f"📂 加载数据: {file_path}")
    df = pd.read_csv(file_path, parse_dates=["date"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].values)
    print(f"✅ 数据预处理完成，标准化后形状: {X_scaled.shape}")
    return X_scaled, scaler, df["date"].values


def train_test_split(data, dates, split_ratio):
    """
    随机划分训练集和测试集

    Args:
        data: 预处理后的数据
        dates: 对应的日期数组
        split_ratio: 划分比例
        is_many_shot: 是否为多样本训练模式

    Returns:
        训练数据，测试数据，训练日期，测试日期
    """
    n_samples = len(data)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    train_size = int(n_samples * split_ratio)
    tr_idx = indices[:]
    te_idx = indices[train_size:]
    return data[tr_idx], data[te_idx], dates[tr_idx], dates[te_idx]


def create_data_loaders(
    train_data, test_data, input_window, pred_window, batch_size=64
):
    """
    创建数据加载器

    Args:
        train_data: 训练数据
        test_data: 测试数据
        input_window: 输入窗口大小
        pred_window: 预测窗口大小
        batch_size: 批次大小

    Returns:
        训练数据加载器，测试数据加载器
    """
    train_dataset = TimeSeriesDataset(train_data, input_window, pred_window)
    test_dataset = TimeSeriesDataset(test_data, input_window, pred_window)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_dataset, test_dataset


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
    test_data,
    test_dates,
    input_window,
    pred_window,
    device,
    scaler,
    noise_levels=[0.01, 0.02, 0.03, 0.04],
    batch_size=64,
    save_dir=None,
):
    """噪声鲁棒性测试"""
    results = []

    for noise_level in noise_levels:
        print(f"\n🧪 噪声水平 {noise_level} 的鲁棒性测试")

        # 添加噪声到测试数据
        test_data_noisy = test_data.copy()
        test_data_noisy += np.random.normal(0, noise_level, test_data.shape)

        # 创建带噪声的测试加载器
        test_dataset = TimeSeriesDataset(test_data_noisy, input_window, pred_window)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 评估在噪声数据上的性能
        mse, mae, preds, targets = evaluate(model, test_loader, device)

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

    data, scaler, dates = load_and_preprocess(cfg["csv"])
    train_data, test_data, train_dates, test_dates = train_test_split(
        data, dates, split_ratio=cfg["train_rate"]
    )
    tr_dl, te_dl, _, _ = create_data_loaders(
        train_data, test_data, cfg["input_len"], cfg["output_len"], cfg["batch"]
    )
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
            avg_loss = train_epoch(tr_dl, epoch, ndp_wrap, opt, device, lr_sched)
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
            test_data,
            test_dates,
            cfg["input_len"],
            cfg["output_len"],
            device,
            scaler,
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
