# src/data.py
from __future__ import annotations
import pandas as pd, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

# ------------------ 目标归一化常量 ------------------
RSRP_MIN = -134.0  # dBm，下限（略小于数据极值）
RSRP_MAX = -57.0  # dBm，上限
THRESH_NORM = 0.5  # 弱覆盖阈值  (≈ -85 dBm)


def rsrp_norm(y: torch.Tensor) -> torch.Tensor:
    return (y - RSRP_MIN) / (RSRP_MAX - RSRP_MIN)


def rsrp_denorm(y_norm: torch.Tensor) -> torch.Tensor:
    return y_norm * (RSRP_MAX - RSRP_MIN) + RSRP_MIN


# ---------------------------------------------------


class RSMap(Dataset):
    """
    每条样本 = 单个栅格测点的 {特征向量, 归一化RSRP}
    由 DataLoader(batch_size=N) 组合 -> [N,D] / [N,1]
    """

    def __init__(self, csv: Path, norm: bool = True):
        df = pd.read_csv(csv)
        df = add_features(df)
        self.df = df
        self.norm = norm
        self.exclude_cols = {
            # 直接排除
            "Cell Index",
            "Frequency Band",
            # 用来计算派生特征的原始几何 / 天线列
            "Cell X",
            "Cell Y",
            "Cell Altitude",
            "Azimuth",
            "Electrical Downtilt",
            "Mechanical Downtilt",
            "RS Power",
            # 原生离散列已被 one‑hot，可一起去掉
            "Clutter Index",
            "Cell Clutter Index",
        }

        # --- 连续特征均值/方差 (供 Z-score) ---
        self.cont_cols = ["rel_alt"]
        self.cont_stats = {c: (df[c].mean(), df[c].std()) for c in self.cont_cols}

        # --- 构建 one-hot 离散列 ---
        df_ohe = pd.get_dummies(df["Clutter Index"].astype(int), prefix="cl")
        df_cell = pd.get_dummies(df["Cell Clutter Index"].astype(int), prefix="cell_cl")
        self.df = pd.concat([df, df_ohe, df_cell], axis=1)

        # 最终特征列 = 连续 + one-hot
        self.x_cols = [
            c
            for c in self.df.columns
            if c.upper() != "RSRP" and c not in self.exclude_cols
        ]

    # ---------------- Dataset API -----------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # 连续特征
        x = row[self.x_cols].values.astype(np.float32)
        # Z-score for rel_alt
        for i, c in enumerate(self.x_cols):
            if c in self.cont_stats:
                mu, std = self.cont_stats[c]
                x[i] = (x[i] - mu) / (std + 1e-6)
        x = torch.tensor(x)

        # 目标归一化
        y = torch.tensor([row["RSRP"]], dtype=torch.float32)
        if self.norm:
            y = rsrp_norm(y)
        return x, y


# ----------------- DataLoader helpers -----------------
def create_loader(csv_path: str, batch: int, N: int):
    """保持旧接口，batch=样本批大小，N 已不再使用"""
    ds = RSMap(Path(csv_path))
    return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)


def split_loaders(csv_path: str, batch: int, split: float = 0.8, seed: int = 42):
    ds = RSMap(Path(csv_path))
    n_train = int(len(ds) * split)
    n_test = len(ds) - n_train
    tr_ds, te_ds = random_split(
        ds, [n_train, n_test], generator=torch.Generator().manual_seed(seed)
    )
    tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True, drop_last=True)
    te_dl = DataLoader(te_ds, batch_size=batch, shuffle=False, drop_last=True)
    return tr_dl, te_dl


# ------------------- Feature Engineering -------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """新增特征 + 连续特征初步缩放"""
    # (1) 几何
    dx = df["X"] - df["Cell X"]
    dy = df["Y"] - df["Cell Y"]
    dist = np.sqrt(dx**2 + dy**2)
    df["log_dist"] = np.log10(dist + 1) / 5  # 0~1 近似
    df["rel_alt"] = df["Altitude"] - df["Cell Altitude"]

    # (2) 角度差
    bearing = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    df["az_error"] = np.abs(bearing - df["Azimuth"]) / 180.0  # 0~1

    # (3) 下倾
    df["tilt_total"] = (
        df["Electrical Downtilt"] + df["Mechanical Downtilt"]
    ) / 90  # 0~1

    # (4) 频率 GHz
    df["freq_GHz"] = df["Frequency Band"] / 6000.0  # 假设 <=6 GHz

    # (5) 发射功率 线性→log10→缩放
    df["rs_pwr_dbw"] = 10 ** (df["RS Power"] / 10) + 1e-9
    df["rs_pwr_dbw"] = np.log10(df["rs_pwr_dbw"]) / 5  # ≈0~1

    # 距离原始坐标、功率等列如需保留可不删除
    return df
