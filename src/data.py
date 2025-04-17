# src/data.py
from __future__ import annotations
import pandas as pd, torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path


# ------------------ 数据集定义 ------------------
class RSMap(Dataset):
    """
    加载预处理后的 CSV 文件：特征已归一化，目标为 RSRP ∈ [0,1]
    每条样本 = {x: [D], y: [1]}
    """

    def __init__(self, csv: Path):
        self.df = pd.read_csv(csv)
        self.x_cols = [c for c in self.df.columns if c.upper() != "RSRP"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.x_cols].values.astype("float32"))
        y = torch.tensor([row["RSRP"]], dtype=torch.float32)
        return x, y


# ------------------ Loader 构建函数 ------------------
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
