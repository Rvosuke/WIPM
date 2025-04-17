# scripts/visualize_rsrp.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import argparse
import torch
import numpy as np

from src.ndp import NDP


def load_model(cfg_path: str, ckpt_path: str, device: str = "cpu"):
    import yaml

    cfg = yaml.safe_load(open(cfg_path))
    model_wrap = NDP(
        in_dim=cfg["D"],
        time_step=cfg["T"],
        device=device,
        hidden=cfg["hidden"],
        n_layers=cfg["layers"],
    )
    state_dict = torch.load(ckpt_path, map_location=device)
    model_wrap.model.load_state_dict(state_dict)
    model_wrap.model.eval()
    return model_wrap


def visualize_rsrp_map(
    csv_path: str,
    cfg_path: str = None,
    ckpt_path: str = None,
    title: str = None,
    save_path: str = None,
    show_residual: bool = False,
):
    df = pd.read_csv(csv_path)
    required_cols = {"X", "Y", "RSRP"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"❌ CSV中缺失以下列：{missing}")

    pivot_true = df.pivot_table(index="Y", columns="X", values="RSRP")

    if not cfg_path or not ckpt_path:
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            pivot_true.sort_index(ascending=False),
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": "RSRP (Normalized)"},
        )
        plt.title(title or Path(csv_path).stem)
        plt.axis("off")
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            print(f"✅ 热力图已保存至: {save_path}")
        else:
            plt.show()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg_path, ckpt_path, device)

    x_cols = [c for c in df.columns if c.upper() != "RSRP"]
    x_tensor = torch.tensor(df[x_cols].values.astype("float32")).to(device)

    with torch.no_grad():
        y_pred = model.sample(x_tensor).squeeze().cpu()
        y_true = torch.tensor(df["RSRP"].values)

    df["RSRP_PRED"] = y_pred
    df["RESIDUAL"] = (y_pred - y_true).abs()

    pivot_pred = df.pivot_table(index="Y", columns="X", values="RSRP_PRED")
    pivot_res = df.pivot_table(index="Y", columns="X", values="RESIDUAL")

    fig, axes = plt.subplots(1, 3 if show_residual else 2, figsize=(18, 5))
    sns.heatmap(
        pivot_true.sort_index(ascending=False),
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        ax=axes[0],
        cbar_kws={"label": "RSRP"},
    )
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    sns.heatmap(
        pivot_pred.sort_index(ascending=False),
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        ax=axes[1],
        cbar_kws={"label": "Predicted RSRP"},
    )
    axes[1].set_title("Model Prediction")
    axes[1].axis("off")

    if show_residual:
        white_red = LinearSegmentedColormap.from_list(
            "whitered",
            ["white", "white", "cyan", "cyan", "blue", "blue", "orange", "red"],
        )
        sns.heatmap(
            pivot_res.sort_index(ascending=False),
            cmap=white_red,
            vmin=0.0,
            vmax=1.0,
            ax=axes[2],
            cbar_kws={"label": "|Error|"},
        )
        axes[2].set_title("Residual (Abs Error)")
        axes[2].axis("off")

    plt.suptitle(title or Path(csv_path).stem)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ 图像已保存至: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="datasets/processed/train_2304601.csv",
        help="输入CSV路径, 需包含X/Y/RSRP列",
    )
    parser.add_argument(
        "--cfg", default="configs/base.yaml", help="模型配置文件 (yaml)"
    )
    parser.add_argument(
        "--ckpt", default="results/checkpoints/ndp_best.pt", help="模型权重文件路径"
    )
    parser.add_argument("--title", default="RSRP Heatmap", help="热力图标题")
    parser.add_argument("--save", default="results/runs", help="保存图像文件路径")
    parser.add_argument("--residual", action="store_true", help="是否显示残差图")
    args = parser.parse_args()
    visualize_rsrp_map(
        args.csv, args.cfg, args.ckpt, args.title, args.save, args.residual
    )
