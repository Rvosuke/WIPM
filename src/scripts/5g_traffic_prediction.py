#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5G Traffic 时间序列预测实验脚本
用于执行基于NDP模型的5G流量数据预测实验，支持多种实验设置。
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.ndp import NDP


class TimeSeriesDataset(Dataset):
    """时间序列数据集类，处理5G流量数据"""

    def __init__(self, data, input_window=12, pred_window=24, stride=1):
        """
        初始化时间序列数据集

        参数:
            data: 原始时间序列数据（已归一化）
            input_window: 输入窗口大小（默认12小时）
            pred_window: 预测窗口大小（长期预测24小时，短期预测6小时）
            stride: 滑动窗口步长（默认为1）
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
    加载并预处理5G流量数据

    参数:
        file_path: CSV文件路径
        feature_cols: 使用的特征列

    返回:
        预处理后的数据，特征缩放器
    """
    print(f"📂 加载数据: {file_path}")
    df = pd.read_csv(file_path, parse_dates=["date"])

    # 检查数据完整性
    print(f"📊 数据形状: {df.shape}, 特征: {feature_cols}")
    print(f"🔍 数据摘要:\n{df[feature_cols].describe()}")

    # 提取特征
    X = df[feature_cols].values

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"✅ 数据预处理完成，标准化后形状: {X_scaled.shape}")
    return X_scaled, scaler, df["date"].values


def train_test_split(data, dates, split_ratio=0.8, is_many_shot=True, random_seed=42):
    """
    随机划分训练集和测试集

    参数:
        data: 预处理后的数据
        dates: 对应的日期数组
        split_ratio: 划分比例
        is_many_shot: 是否为多样本训练模式
        random_seed: 随机种子，确保结果可复现

    返回:
        训练数据，测试数据，训练日期，测试日期
    """
    # 设置随机种子确保结果可复现
    np.random.seed(random_seed)

    # 获取数据总数
    n_samples = len(data)

    # 创建索引并随机打乱
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # 根据是否为多样本训练确定训练集大小
    if is_many_shot:
        # 多样本训练: 80%训练，20%测试
        train_size = int(n_samples * split_ratio)
    else:
        # 少样本训练: 20%训练，80%测试
        train_size = int(n_samples * (1 - split_ratio))

    # 划分训练集和测试集索引
    train_indices = indices[:]
    test_indices = indices[train_size:]

    # 根据索引获取相应的数据
    train_data = data[train_indices]
    test_data = data[test_indices]
    train_dates = dates[train_indices]
    test_dates = dates[test_indices]

    # print(f"📊 数据集随机划分: 训练集 {train_data.shape}, 测试集 {test_data.shape}")
    return train_data, test_data, train_dates, test_dates


def create_data_loaders(
    train_data, test_data, input_window, pred_window, batch_size=64
):
    """
    创建数据加载器

    参数:
        train_data: 训练数据
        test_data: 测试数据
        input_window: 输入窗口大小
        pred_window: 预测窗口大小
        batch_size: 批次大小

    返回:
        训练数据加载器，测试数据加载器
    """
    train_dataset = TimeSeriesDataset(train_data, input_window, pred_window)
    test_dataset = TimeSeriesDataset(test_data, input_window, pred_window)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(model, train_loader, optimizer, device, epoch, writer=None):
    """训练一个epoch"""
    model.model.train()
    total_loss = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # 直接计算整个序列的损失
        # x形状：[B, input_window, feature_dim]
        # y形状：[B, pred_window, 1]
        loss = model.loss(x, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    if writer:
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"训练 Epoch: {epoch} \t损失: {avg_loss:.6f}")
    return avg_loss


@torch.no_grad()
def evaluate(model, test_loader, device, scaler, epoch, writer=None):
    """评估模型性能"""
    model.model.eval()
    all_preds = []
    all_targets = []

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        # 使用NDP模型直接预测整个序列
        # x形状：[B, input_window, feature_dim]
        # y形状：[B, pred_window, 1]
        pred_seq_len = y.size(1)
        y_pred = model.sample_sequence(x, pred_seq_len)

        # 收集预测和目标值
        all_preds.append(y_pred)
        all_targets.append(y)

    # 合并批次结果
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 直接使用torch计算评估指标，不进行反标准化
    mse = torch.mean((all_preds - all_targets) ** 2).item()
    mae = torch.mean(torch.abs(all_preds - all_targets)).item()

    if writer:
        writer.add_scalar("MSE/val", mse, epoch)
        writer.add_scalar("MAE/val", mae, epoch)

    print(f"评估结果 - MSE: {mse:.4f}, MAE: {mae:.4f}")

    # 返回numpy格式用于可视化
    return mse, mae, all_preds.cpu().numpy(), all_targets.cpu().numpy()


def visualize_predictions(preds, targets, dates, save_path, title):
    """可视化预测结果"""
    plt.figure(figsize=(15, 6))

    # 如果预测包含多个序列，我们绘制第一个序列作为示例
    if len(preds.shape) == 3:
        preds = preds[0]
        targets = targets[0]
        vis_dates = dates[: len(targets)]
    else:
        vis_dates = dates[: len(targets)]

    plt.plot(vis_dates, targets, "b-", label="真实值")
    plt.plot(vis_dates, preds, "r--", label="预测值")

    plt.title(title)
    plt.xlabel("日期")
    plt.ylabel("流量值")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(save_path)
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
        mse, mae, preds, targets = evaluate(model, test_loader, device, scaler, 0)

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
        plt.title("MSE在不同噪声水平下的性能")
        plt.xlabel("噪声水平")
        plt.ylabel("MSE")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(noise_levels, mae_values, "s-", color="red")
        plt.title("MAE在不同噪声水平下的性能")
        plt.xlabel("噪声水平")
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


def run_experiment(
    data_path,
    save_dir=None,
    is_many_shot=True,
    is_long_term=True,
    input_window=12,
    hidden_dim=128,
    n_layers=6,
    time_steps=25,
    epochs=100,
    batch_size=64,
    learning_rate=3e-4,
    device=None,
):
    """运行完整的实验流程"""
    # 设置设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用设备: {device}")

    # 创建保存目录
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        data_name = Path(data_path).stem
        experiment_type = "many" if is_many_shot else "few"
        pred_type = "long" if is_long_term else "short"
        save_dir = Path(
            f"results/5g_traffic/{data_name}_{experiment_type}_{pred_type}_{timestamp}"
        )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = save_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    tb_dir = save_dir / "runs"
    tb_dir.mkdir(exist_ok=True)

    # 设置TensorBoard
    writer = SummaryWriter(tb_dir)

    # 设置预测窗口大小
    pred_window = 24 if is_long_term else 6

    # 加载和预处理数据
    data, scaler, dates = load_and_preprocess(data_path)

    # 划分训练集和测试集
    train_data, test_data, train_dates, test_dates = train_test_split(
        data, dates, split_ratio=0.8, is_many_shot=is_many_shot
    )

    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(
        train_data, test_data, input_window, pred_window, batch_size
    )

    # 初始化模型
    input_dim = train_data.shape[1]  # 特征维度
    model = NDP(
        in_dim=input_dim,
        time_step=time_steps,
        device=device,
        hidden=hidden_dim,
        n_layers=n_layers,
    )

    # 设置优化器
    optimizer = torch.optim.AdamW(
        model.model.parameters(), lr=learning_rate, weight_decay=1e-2
    )

    # 记录配置信息
    config = {
        "data_path": data_path,
        "is_many_shot": is_many_shot,
        "is_long_term": is_long_term,
        "input_window": input_window,
        "pred_window": pred_window,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "time_steps": time_steps,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }

    # 保存配置信息
    with open(save_dir / "config.txt", "w") as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

    # 训练循环
    best_mse = float("inf")
    train_losses = []
    val_mses = []
    val_maes = []

    print(
        f"\n🚀 开始训练 - {'多样本' if is_many_shot else '少样本'} {'长期' if is_long_term else '短期'}预测"
    )
    for epoch in range(1, epochs + 1):
        # 训练
        train_loss = train(model, train_loader, optimizer, device, epoch, writer)
        train_losses.append(train_loss)

        # 评估
        mse, mae, _, _ = evaluate(model, test_loader, device, scaler, epoch, writer)
        val_mses.append(mse)
        val_maes.append(mae)

        # 保存最佳模型
        if mse < best_mse:
            best_mse = mse
            torch.save(model.model.state_dict(), checkpoints_dir / "ndp_best.pt")
            print(f"💾 保存最佳模型 (MSE: {best_mse:.4f})")

        # 每25个epoch保存一次检查点
        if epoch % 25 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_mse": mse,
                    "val_mae": mae,
                },
                checkpoints_dir / f"ndp_epoch_{epoch}.pt",
            )

    # 加载最佳模型
    model.model.load_state_dict(
        torch.load(checkpoints_dir / "ndp_best.pt", map_location=device)
    )

    # 最终评估
    mse, mae, preds, targets = evaluate(model, test_loader, device, scaler, epochs + 1)

    # 可视化预测结果
    # visualize_predictions(
    #     preds,
    #     targets,
    #     test_dates,
    #     save_dir / "final_prediction.png",
    #     f"最终预测结果 ({'多样本' if is_many_shot else '少样本'} {'长期' if is_long_term else '短期'})",
    # )

    # 绘制训练曲线
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(range(1, epochs + 1), train_losses)
    plt.title("训练损失")
    plt.xlabel("Epoch")
    plt.ylabel("损失")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(range(1, epochs + 1), val_mses, "r-")
    plt.title("验证 MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(range(1, epochs + 1), val_maes, "g-")
    plt.title("验证 MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png")

    # 进行噪声鲁棒性测试
    noise_results = noise_robustness_test(
        model,
        test_data,
        test_dates,
        input_window,
        pred_window,
        device,
        scaler,
        save_dir=save_dir,
    )

    # 保存最终结果
    final_results = {
        "best_mse": best_mse,
        "final_mse": mse,
        "final_mae": mae,
    }

    with open(save_dir / "final_results.txt", "w") as f:
        f.write("最终实验结果:\n")
        for k, v in final_results.items():
            f.write(f"{k}: {v}\n")

    print(f"\n✅ 实验完成! 结果已保存到 {save_dir}")
    return model, final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5G流量预测实验")
    parser.add_argument("--data", type=str, required=True, help="数据文件路径")
    parser.add_argument("--save_dir", type=str, default=None, help="结果保存目录")
    parser.add_argument(
        "--many_shot", action="store_true", help="使用多样本训练模式(80%训练)"
    )
    parser.add_argument(
        "--few_shot", action="store_true", help="使用少样本训练模式(20%训练)"
    )
    parser.add_argument("--long_term", action="store_true", help="长期预测(24小时)")
    parser.add_argument("--short_term", action="store_true", help="短期预测(6小时)")
    parser.add_argument(
        "--input_window", type=int, default=12, help="输入窗口大小(小时)"
    )
    parser.add_argument("--hidden_dim", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--n_layers", type=int, default=6, help="网络层数")
    parser.add_argument("--time_steps", type=int, default=25, help="扩散步数")
    parser.add_argument("--epochs", type=int, default=16, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-7, help="学习率")
    parser.add_argument("--cuda", type=int, default=0, help="使用的CUDA设备ID")

    args = parser.parse_args()

    # 设置CUDA设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cpu")

    # 确定训练模式
    if args.many_shot and args.few_shot:
        raise ValueError("不能同时指定--many_shot和--few_shot")
    elif not (args.many_shot or args.few_shot):
        args.many_shot = True  # 默认使用多样本训练

    is_many_shot = args.many_shot

    # 确定预测模式
    if args.long_term and args.short_term:
        raise ValueError("不能同时指定--long_term和--short_term")
    elif not (args.long_term or args.short_term):
        args.long_term = True  # 默认使用长期预测

    is_long_term = args.long_term

    # 运行实验
    model, results = run_experiment(
        data_path=args.data,
        save_dir=args.save_dir,
        is_many_shot=is_many_shot,
        is_long_term=is_long_term,
        input_window=args.input_window,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        time_steps=args.time_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
    )
