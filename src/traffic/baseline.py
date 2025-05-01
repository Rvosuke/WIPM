#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5G流量预测 - 基线模型比较脚本
比较线性回归和SVR与NDP模型在5G流量预测任务上的性能
"""

import os, sys, argparse, torch, yaml
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).resolve().parents[2]))
np.random.seed(2025)

from src.ndp import NDP
from src.traffic.main import (
    load_and_preprocess,
    train_test_split,
    create_data_loaders,
    evaluate,
)


class BaselineModel:
    def __init__(self, name):
        self.name = name

    def train(self, train_data):
        print(f"🏋️‍♀️ 正在训练 {self.name} 模型...")
        X_train, y_train = [], []
        for x, y in train_data:
            X_train.append(x.reshape(-1).numpy())
            y_train.append(y.squeeze(-1).numpy())
        X_train = np.vstack(X_train)
        y_train = np.vstack(y_train)
        self.model.fit(X_train, y_train)
        print(f"✅ {self.name} 模型训练完成")

    def evaluate(self, test_loader):
        print(f"🔍 评估 {self.name} 模型...")
        X_test, y_test = [], []
        for x, y in test_loader:
            X_test.append(x.reshape(-1).numpy())
            y_test.append(y.squeeze(-1).numpy())
        X_test = np.vstack(X_test)
        y_test = np.vstack(y_test)
        y_pred = self.model.predict(X_test)
        mse = np.mean((y_pred - y_test) ** 2)
        mae = np.mean(np.abs(y_pred - y_test))
        print(f"[{self.name}] MSE: {mse:.4f}, MAE: {mae:.4f}")
        return mse, mae, y_pred, y_test

    def interface(self, x):
        pass


class LinearModel(BaselineModel):
    def __init__(self, name="linear"):
        super().__init__(name)
        self.model = LinearRegression()


class SVRModel(BaselineModel):
    def __init__(self, name="SVR", kernel="rbf", C=1.0, epsilon=0.1, gamma="scale"):
        super().__init__(name)
        svr = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        self.model = MultiOutputRegressor(svr)


def visualize_model_predictions(model_results, save_dir, transfer=False):
    """可视化不同模型的预测结果对比"""
    plt.figure(figsize=(15, 8))

    # 选取一个样本进行可视化
    sample_idx = 10

    # 获取真实值
    if transfer:
        targets = next(iter(model_results.values()))["trans_targ"][sample_idx]
    else:
        targets = next(iter(model_results.values()))["targets"][sample_idx]

    # 设置x轴
    x_points = np.arange(len(targets))

    # 绘制真实值
    plt.plot(x_points, targets, "k-", linewidth=2, label="Ground Truth")

    # 绘制每个模型的预测值
    colors = ["b", "r", "g"]
    for i, (model_name, results) in enumerate(model_results.items()):
        if transfer:
            predictions = results["trans_pre"][sample_idx]
        else:
            predictions = results["predictions"][sample_idx]

        plt.plot(
            x_points,
            predictions,
            f"{colors[i]}--",
            linewidth=2,
            label=f"{model_name} Prediction",
        )

    # 添加标题和标签
    if transfer:
        save_path = save_dir / "transfer_comparison.png"
        plt.title("Transfer Comparison", fontsize=16)
    else:
        save_path = save_dir / "model_comparison.png"
        plt.title("Baseline Comparison", fontsize=16)
    plt.xlabel("Time Step", fontsize=14)
    plt.ylabel("Prediction", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # 保存图表
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"📊 模型预测结果对比已保存至: {save_path}")


def create_performance_table(model_results, save_dir):
    """创建性能对比表格并保存"""
    # 准备表格数据
    models = list(model_results.keys())
    metrics = ["MSE", "MAE"]
    data = {
        "Model": models,
        "MSE": [model_results[model]["mse"] for model in models],
        "MAE": [model_results[model]["mae"] for model in models],
        "Transfer MSE": [model_results[model]["trans_mse"] for model in models],
        "Transfer MAE": [model_results[model]["trans_mae"] for model in models],
    }

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 保存为CSV
    save_path = save_dir / "performance_comparison.csv"
    df.to_csv(save_path, index=False)

    # 保存为文本文件，更易读
    with open(save_dir / "performance_comparison.txt", "w") as f:
        f.write("模型性能对比:\n")
        f.write("=" * 40 + "\n")
        for model in models:
            f.write(f"{model}:\n")
            f.write(f"  MSE: {model_results[model]['mse']:.4f}\n")
            f.write(f"  MAE: {model_results[model]['mae']:.4f}\n")
            f.write(f"  Transfer MSE: {model_results[model]['trans_mse']:.4f}\n")
            f.write(f"  Transfer MAE: {model_results[model]['trans_mae']:.4f}\n")
            f.write("-" * 40 + "\n")

    print(f"📋 模型性能对比表已保存至: {save_path}")


def run_baseline_comparison(cfg):
    """运行基线模型比较实验"""
    # 设置设备
    device = torch.device(cfg["device"])
    data_name = Path(cfg["csv"]).stem
    task_name = data_name / Path(f"out{cfg["output_len"]}_rate{cfg["train_rate"]}")
    save_dir = Path(cfg["save_dir"]) / task_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载和预处理数据
    data, _, dates = load_and_preprocess(cfg["csv"])
    train_data, test_data, _, _ = train_test_split(
        data, dates, split_ratio=cfg["train_rate"]
    )
    train_loader, test_loader, train_dataset, test_dataset = create_data_loaders(
        train_data, test_data, cfg["input_len"], cfg["output_len"], cfg["batch"]
    )

    transfer_data, _, datas = load_and_preprocess(cfg["transfer_csv"])
    _, transfer_data, _, _ = train_test_split(transfer_data, datas, 0)
    _, transfer_loader, _, transfer_dataset = create_data_loaders(
        transfer_data, transfer_data, cfg["input_len"], cfg["output_len"], cfg["batch"]
    )

    # 初始化模型
    models = {
        "Linear": LinearModel(),
        "SVR": SVRModel(),
        "NDP": NDP(
            cfg["D"],
            cfg["T"],
            device,
            hidden=cfg["hidden"],
            n_layers=cfg["layers"],
            in_len=cfg["input_len"],
            out_len=cfg["output_len"],
        ),
    }

    # 存储结果
    model_results = {}

    # 训练和评估每个模型
    for name, model in models.items():
        if name in ["Linear", "SVR"]:
            model.train(train_dataset)
            mse, mae, predictions, targets = model.evaluate(test_dataset)
            trans_mse, trans_mae, trans_pre, trans_targ = model.evaluate(
                transfer_dataset
            )
        else:  # NDP模型
            model.model.load_state_dict(
                torch.load(
                    f"results/5g_traffic/{task_name}/checkpoints/ndp_best.pt",
                    map_location=device,
                )
            )
            # 获取最佳性能
            print("评估NDP模型")
            mse, mae, predictions, targets = evaluate(model, test_loader, device)
            predictions = predictions.squeeze(-1)
            targets = targets.squeeze(-1)
            trans_mse, trans_mae, trans_pre, trans_targ = evaluate(
                model, transfer_loader, device
            )
            trans_pre = trans_pre.squeeze(-1)
            trans_targ = trans_targ.squeeze(-1)

        # 保存结果
        model_results[name] = {
            "mse": mse,
            "mae": mae,
            "predictions": predictions,
            "targets": targets,
            "trans_mse": trans_mse,
            "trans_mae": trans_mae,
            "trans_pre": trans_pre,
            "trans_targ": trans_targ,
        }

    # 可视化不同模型的预测结果
    visualize_model_predictions(model_results, save_dir)
    visualize_model_predictions(model_results, save_dir, True)

    # 创建并保存性能对比表格
    create_performance_table(model_results, save_dir)

    print(f"\n✅ 基线模型比较实验完成! 结果已保存到 {save_dir}")

    return model_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5G流量预测基线模型比较实验")
    parser.add_argument("--cfg", default="configs/traffic_baseline.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    run_baseline_comparison(cfg)
