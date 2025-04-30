#!/usr/bin/env python
# src/beselines/baseline_transfer.py
"""
基线模型迁移学习脚本 - 使用两个数据集作为训练集，一个数据集作为测试集
与NDP模型的零样本迁移学习能力进行对比
"""

import torch, yaml, argparse, sys, warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

sys.path.append(str(Path(__file__).parents[2]))
from src.data import RSMap
from src.metrics import rmse, pcrr
from src.baselines.baseline_models import LinearModel, SVRModel, BaselineModel
from src.scripts.visualize_rsrp import visualize_rsrp_map


def load_multiple_datasets(csv_paths):
    """加载并合并多个CSV数据集"""
    dataframes = []
    for path in csv_paths:
        print(f"📊 加载数据集: {path}")
        dataset = RSMap(Path(path))
        dataframes.append(dataset.df)

    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"✓ 合并后数据集总样本数: {len(merged_df)}")
    return merged_df


def prepare_data(train_dfs, test_df):
    """准备训练集和测试集数据"""
    # 合并训练数据框
    if isinstance(train_dfs, list):
        train_df = pd.concat(train_dfs, ignore_index=True)
    else:
        train_df = train_dfs

    # 提取特征和目标值
    x_cols = [c for c in train_df.columns if c.upper() != "RSRP"]
    X_train = train_df[x_cols].values
    y_train = train_df["RSRP"].values.reshape(-1, 1)

    X_test = test_df[x_cols].values
    y_test = test_df["RSRP"].values.reshape(-1, 1)

    return X_train, X_test, y_train, y_test


def train_baselines_transfer(cfg):
    """训练和评估基线模型的迁移学习性能"""
    # 加载训练和测试数据集
    print(f"🔄 加载训练数据集...")
    train_dataframes = [RSMap(Path(csv_path)).df for csv_path in cfg["train_csv_paths"]]

    print(f"🔄 加载测试数据集: {cfg['test_csv_path']}")
    test_df = RSMap(Path(cfg["test_csv_path"])).df

    # 准备数据
    X_train, X_test, y_train, y_test = prepare_data(train_dataframes, test_df)
    print(f"✓ 训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

    # 创建模型字典
    models = {
        "Linear": LinearModel(
            {
                "fit_intercept": cfg.get("linear_fit_intercept", True),
            }
        ),
        "SVR": SVRModel(
            {
                "kernel": cfg.get("svr_kernel", "rbf"),
                "C": cfg.get("svr_C", 1.0),
                "epsilon": cfg.get("svr_epsilon", 0.1),
                "gamma": cfg.get("svr_gamma", "scale"),
            }
        ),
    }

    results = {}
    save_dir = Path(cfg.get("save_dir", "results/transfer_baselines"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # 训练和评估每个模型
    for name, model in models.items():
        print(f"🏋️‍♀️ 正在训练 {name} 迁移学习模型...")
        model.train(X_train, y_train)

        print(f"📊 评估 {name} 迁移学习模型性能...")
        metrics = model.evaluate(X_test, y_test)
        results[name] = metrics

        print(f"[{name}] RMSE={metrics['rmse']:.4f} | PCRR={metrics['pcrr']:.4f}")

        # 为每个模型生成可视化结果
        if cfg.get("visualize", False):
            vis_save = save_dir / f"{name.lower()}_transfer_prediction.png"

            print(f"🎨 生成 {name} 模型的迁移学习RSRP预测可视化...")
            model.visualize_prediction(
                cfg["test_csv_path"],
                title=f"{name} - Transfer Learning on {Path(cfg['test_csv_path']).stem}",
                save_path=vis_save,
                show_residual=cfg.get("show_residual", True),
            )

    # 加载NDP迁移学习结果进行对比(如果存在)
    ndp_transfer_file = Path(
        cfg.get("ndp_transfer_results", "results/transfer/transfer_results.txt")
    )
    ndp_metrics = None

    if ndp_transfer_file.exists():
        print(f"📈 加载NDP迁移学习结果: {ndp_transfer_file}")
        ndp_metrics = {"rmse": None, "pcrr": None}

        with open(ndp_transfer_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "最终测试 RMSE:" in line:
                    ndp_metrics["rmse"] = float(line.split(":")[-1].strip())
                elif "最终测试 PCRR:" in line:
                    ndp_metrics["pcrr"] = float(line.split(":")[-1].strip())

        if ndp_metrics["rmse"] is not None:
            print(
                f"[NDP] RMSE={ndp_metrics['rmse']:.4f} | PCRR={ndp_metrics['pcrr']:.4f}"
            )
            results["NDP"] = ndp_metrics

    # 绘制对比图
    plot_comparison(results, save_dir)

    # 保存结果
    with open(save_dir / "baseline_transfer_results.txt", "w") as f:
        f.write("迁移学习模型性能对比:\n")
        f.write("=" * 40 + "\n")
        for name, metrics in results.items():
            f.write(f"{name}:\n")
            f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
            f.write(f"  PCRR: {metrics['pcrr']:.4f}\n")
            f.write("-" * 40 + "\n")

    print(f"✅ 基线模型迁移学习评估完成！结果已保存至: {save_dir}")
    return results


def plot_comparison(results, save_dir):
    """绘制基线模型与NDP模型的迁移学习性能对比图"""
    # 准备数据
    models = list(results.keys())
    rmse_values = [results[m]["rmse"] for m in models]
    pcrr_values = [results[m]["pcrr"] for m in models]

    # 设置颜色
    colors = (
        ["#5DA5DA", "#FAA43A", "#60BD68"]
        if len(models) == 3
        else ["#5DA5DA", "#FAA43A"]
    )

    # 绘制对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # RMSE 对比 (越低越好)
    bars1 = ax1.bar(models, rmse_values, color=colors)
    ax1.set_title("RMSE Comparison (Transfer Learning)")
    ax1.set_ylabel("RMSE")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # 在柱状图上标注具体数值
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    # PCRR 对比 (越高越好)
    bars2 = ax2.bar(models, pcrr_values, color=colors)
    ax2.set_title("PCRR Comparison (Transfer Learning)")
    ax2.set_ylabel("PCRR")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # 在柱状图上标注具体数值
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    plt.suptitle("Transfer Learning Performance Comparison")
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_dir / "transfer_comparison.png")
    plt.close()
    print(f"✅ 性能对比图已保存至: {save_dir / 'transfer_comparison.png'}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="基线模型迁移学习评估脚本")
    parser.add_argument(
        "--cfg", default="configs/baseline_transfer.yaml", help="配置文件路径"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("📊 开始基线模型迁移学习评估...")
    print("=" * 50)

    # 加载配置文件
    cfg = yaml.safe_load(open(args.cfg))

    # 训练和评估基线模型
    train_baselines_transfer(cfg)
