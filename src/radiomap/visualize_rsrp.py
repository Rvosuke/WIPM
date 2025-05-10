# src/radiomap/visualize_rsrp.py
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import torch, yaml, argparse
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, median_filter
import cv2

from src.ndp import NDP


def load_model(cfg: dict, device: str = "cpu"):
    model_wrap = NDP(
        in_dim=cfg["D"],
        time_step=cfg["T"],
        device=device,
        hidden=cfg["hidden"],
        n_layers=cfg["layers"],
    )
    state_dict = torch.load(cfg["ckpt"], map_location=device)
    model_wrap.model.load_state_dict(state_dict)
    model_wrap.model.eval()
    return model_wrap


def interpolate_grid(resolution=100):
    """创建高分辨率网格并准备用于预测的特征"""
    x_grid = np.linspace(0, 1, resolution)
    y_grid = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    return pd.DataFrame({"X": xx.flatten(), "Y": yy.flatten()})


def apply_smoothing(data, sigma=0.5, kernel_size=3):
    """对预测地图应用平滑处理

    Args:
        data: 需要平滑的数据矩阵
        method: 平滑方法，可选 'gaussian', 'median', 'bilateral'
        sigma: 高斯滤波的sigma参数
        kernel_size: 中值滤波的核大小

    Returns:
        平滑处理后的数据矩阵
    """

    data = np.nan_to_num(data, nan=np.nanmean(data))

    # 步骤1: 中值滤波去除离群噪声点
    data = median_filter(data, size=5)

    # 步骤2: 高斯滤波进行初步平滑 (使用较大sigma值)
    data = gaussian_filter(data, sigma=3.0)

    # 步骤3: 归一化到0-255用于双边滤波
    data_norm = (
        (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data)) * 255
    ).astype(np.uint8)

    # 步骤4: 应用多次双边滤波达到更强的平滑效果
    filtered = cv2.bilateralFilter(data_norm, d=9, sigmaColor=100, sigmaSpace=100)
    # 再应用一次双边滤波以增强平滑效果
    filtered = cv2.bilateralFilter(filtered, d=9, sigmaColor=75, sigmaSpace=75)

    # 步骤5: 转换回原始数据范围
    data = filtered / 255 * (np.nanmax(data) - np.nanmin(data)) + np.nanmin(data)

    # 步骤6: 最终使用一次高斯滤波使结果更加平滑
    data = gaussian_filter(data, sigma=2.0)

    return data


def visualize_rsrp_map(
    cfg: dict = None,
    title: str = None,
    resolution: int = 100,
    full_coverage: bool = False,
):
    df = pd.read_csv(cfg["csv"])
    batch_size = cfg.get("batch", 256)
    required_cols = {"X", "Y", "RSRP"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"❌ CSV中缺失以下列 {missing}")

    # 创建原始数据的透视表
    pivot_true = df.pivot_table(index="Y", columns="X", values="RSRP")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device)
    x_cols = [c for c in df.columns if c.upper() != "RSRP"]

    if full_coverage:  # 如果需要全覆盖，则创建高分辨率网格
        print(f"生成 {resolution}x{resolution} 分辨率的完整网格...")
        grid_df = interpolate_grid(resolution)
        predictions = []
        x_tensor = torch.tensor(grid_df[x_cols].values.astype("float32"), device=device)
        with torch.no_grad():
            for i in tqdm(range(0, len(grid_df), batch_size)):
                batch = x_tensor[i : i + batch_size].unsqueeze(1)
                batch_pred = model.sample(batch, 1).squeeze().cpu().numpy()
                predictions.append(batch_pred)
        grid_df["RSRP_PRED"] = np.concatenate(predictions)
        pivot_pred = grid_df.pivot_table(index="Y", columns="X", values="RSRP_PRED")
    else:  # 只对CSV中的点进行预测
        predictions = []
        x_tensor = torch.tensor(df[x_cols].values.astype("float32"), device=device)
        with torch.no_grad():
            for i in tqdm(range(0, len(df), batch_size)):
                batch = x_tensor[i : i + batch_size]
                batch_pred = model.sample(batch, 1).squeeze().cpu().numpy()
                predictions.append(batch_pred)
        df["RSRP_PRED"] = np.concatenate(predictions)
        pivot_pred = df.pivot_table(index="Y", columns="X", values="RSRP_PRED")
    pivot_pred_values = pivot_pred.values
    smoothed_values = apply_smoothing(pivot_pred_values)
    pivot_pred = pd.DataFrame(
        smoothed_values, index=pivot_pred.index, columns=pivot_pred.columns
    )
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    sns.heatmap(  # 绘制真实值热图（仅包含原始数据点）
        pivot_true.sort_index(ascending=False),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        ax=axes[0],
        cbar_kws={"label": "RSRP"},
    )
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    sns.heatmap(  # 绘制预测热图（可能是完整网格）
        pivot_pred.sort_index(ascending=False),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        ax=axes[1],
        cbar_kws={"label": "Predicted RSRP"},
    )
    axes[1].set_title(f"Model Prediction")
    axes[1].axis("off")
    plt.suptitle(title or Path(cfg["csv"]).stem)
    plt.tight_layout()
    save_path = cfg["save"]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"✅ 图像已保存至: {save_path}")

    if full_coverage and save_path:
        results_csv = Path(save_path).with_suffix(".csv")
        grid_df.to_csv(results_csv, index=False)
        print(f"✅ 完整预测结果已保存至: {results_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/base.yaml", help="模型配置文件")
    parser.add_argument("--title", default="RSRP Heatmap", help="热力图标题")
    parser.add_argument("--resolution", type=int, default=360, help="网格分辨率")
    parser.add_argument("--full", action="store_true", default=True, help="是否全覆盖")
    args = parser.parse_args()

    visualize_rsrp_map(
        yaml.safe_load(open(args.cfg)),
        args.title,
        args.resolution,
        args.full,
    )
