# src/scripts/preproces.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# ------- 常量参数 --------
RSRP_MIN = -135.0  # dBm，下限（略小于数据极值）
RSRP_MAX = -57.0  # dBm，上限


def rsrp_norm(val):
    return (val - RSRP_MIN) / (RSRP_MAX - RSRP_MIN)


def rsrp_denorm(y_norm):
    return y_norm * (RSRP_MAX - RSRP_MIN) + RSRP_MIN


def normalize_minmax(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-8)


def normalize_zscore(series):
    return (series - series.mean()) / (series.std() + 1e-8)


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    # 衍生特征
    dx = df["X"] - df["Cell X"]
    dy = df["Y"] - df["Cell Y"]
    dist = np.sqrt(dx**2 + dy**2)
    df["log_dist"] = np.log10(dist + 1)
    df["rel_alt"] = df["Altitude"] - df["Cell Altitude"]

    bearing = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    df["az_error"] = np.abs(bearing - df["Azimuth"]) / 180.0

    df["tilt_total"] = df["Electrical Downtilt"] + df["Mechanical Downtilt"]
    df["rs_pwr_log"] = np.log10(10 ** (df["RS Power"] / 10) + 1e-6)

    # 高度类复合特征
    df["tx_abs_height"] = df["Cell Altitude"] + df["Height"]
    df["rx_abs_height"] = df["Altitude"] + df["Building Height"]
    df["height_diff"] = df["tx_abs_height"] - df["rx_abs_height"]
    df["cell_building_ratio"] = df["Height"] / (df["Cell Building Height"] + 1.0)
    df["user_building_flag"] = (df["Building Height"] > 0).astype(float)

    # 特征归一化
    df["X"] = normalize_minmax(df["X"])
    df["Y"] = normalize_minmax(df["Y"])
    df["log_dist"] = normalize_minmax(df["log_dist"])
    df["rel_alt"] = normalize_zscore(df["rel_alt"])
    df["tilt_total"] = normalize_minmax(df["tilt_total"])
    df["rs_pwr_log"] = normalize_minmax(df["rs_pwr_log"])
    df["tx_abs_height"] = normalize_minmax(df["tx_abs_height"])
    df["rx_abs_height"] = normalize_minmax(df["rx_abs_height"])
    df["height_diff"] = normalize_zscore(df["height_diff"])
    df["cell_building_ratio"] = normalize_minmax(df["cell_building_ratio"])
    df["Height"] = normalize_minmax(df["Height"])

    # 离散 One-hot
    clut = pd.get_dummies(df["Clutter Index"].astype(int), prefix="cl")
    cell_clut = pd.get_dummies(df["Cell Clutter Index"].astype(int), prefix="cell_cl")

    # 移除原始冗余列（除保留必要的 X/Y）
    exclude_cols = {
        "Cell Index",
        "Frequency Band",
        "RS Power",
        "Cell X",
        "Cell Y",
        "Altitude",
        "Cell Altitude",
        "Azimuth",
        "Electrical Downtilt",
        "Mechanical Downtilt",
        "Clutter Index",
        "Cell Clutter Index",
        "Building Height",
        "Cell Building Height",
    }
    df = df.drop(columns=[c for c in exclude_cols if c in df.columns])

    # 归一化 RSRP
    df["RSRP"] = rsrp_norm(df["RSRP"])

    # 拼接 one-hot
    df = pd.concat([df, clut, cell_clut], axis=1)

    return df


def main(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    df = preprocess_df(df)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed CSV saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="datasets/raw/train_468101.csv",
        help="Path to raw CSV",
    )
    parser.add_argument(
        "--output",
        default="datasets/processed/train_468101.csv",
        help="Path to save processed CSV",
    )
    args = parser.parse_args()
    main(args.input, args.output)
