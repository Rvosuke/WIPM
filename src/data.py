# src/data.py
import pandas as pd, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class RSMap(Dataset):
    def __init__(self, csv: Path):
        self.df = pd.read_csv(csv)
        self.x_cols = [c for c in self.df.columns if c.upper() != "RSRP"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.x_cols].values.astype("float32")).unsqueeze(0)
        y = torch.tensor([row["RSRP"]], dtype=torch.float32).unsqueeze(0)
        return x, y


class TrafficSeries(Dataset):
    def __init__(
        self,
        data,
        input_window,
        pred_window,
        split_ratio,
        train=True,
        feature_cols=["value", "ma_value", "lg_ma_value"],
    ):
        """
        初始化时间序列数据集, 处理5G流量数据

        Args:
            data: 原始时间序列数据路径（已归一化）
            input_window: 输入窗口大小
            pred_window: 预测窗口大小
            split_ratio: 训练集与测试集的划分比例
            train: 是否为训练集
            feature_cols: 选择的特征列
        """
        # 读取数据
        df = pd.read_csv(data, parse_dates=["date"])
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(df[feature_cols].values)
        self.date = df["date"].values  # shape[n_samples, n_features]

        # 时间序列需要保证按顺序划分训练集和测试集
        n_samples = len(self.data)
        train_size = int(n_samples * split_ratio)

        # 顺序划分，避免数据泄漏
        # 注意 self.data 与 self.date 均为 numpy.array 数据
        if train:
            self.data = self.data[:train_size]
            self.date = self.date[:train_size]
        else:
            self.data = self.data[train_size:]
            self.date = self.date[train_size:]

        self.input_window = input_window
        self.pred_window = pred_window
        self._prepare_sequences()

    def _prepare_sequences(self):
        """准备输入序列和目标序列"""
        self.samples = []
        total_len = len(self.data)

        # 生成时间序列样本，确保数据不交叉
        for i in range(total_len - self.input_window - self.pred_window + 1):
            x_seq = self.data[i : i + self.input_window]
            y_seq = self.data[
                i + self.input_window : i + self.input_window + self.pred_window, 0:1
            ]  # 只取value列作为预测目标
            self.samples.append((x_seq, y_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """返回形状为[input_window, feature_dim]的输入和形状为[pred_window, 1]的输出"""
        x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)

    def to_numpy(self):
        """
        将数据转化为sklearn兼容的格式
        """
        x_seqs = np.vstack(
            [sample[0].reshape(-1) for sample in self.samples]
        )  # Stack x_seq (input sequences)
        y_seqs = np.vstack(
            [sample[1].reshape(-1) for sample in self.samples]
        )  # Stack y_seq (target sequences)
        return x_seqs, y_seqs


def split_loaders(csv_path: str, batch: int, split: float, seed: int = 42):
    tr_ds = RSMap(Path(csv_path))
    n_train = int(len(tr_ds) * split)
    n_test = len(tr_ds) - n_train
    _, te_ds = random_split(
        tr_ds, [n_train, n_test], generator=torch.Generator().manual_seed(seed)
    )
    tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True, drop_last=True)
    te_dl = DataLoader(te_ds, batch_size=batch, shuffle=False, drop_last=True)
    return tr_dl, te_dl
