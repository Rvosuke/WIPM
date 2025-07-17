# src/radiomap/transfer.py
"""
迁移学习训练脚本 - 使用两个数据集作为训练集，一个数据集作为测试集
用于0样本测试场景
"""

import torch, yaml, argparse, sys, warnings
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset

sys.path.append(str(Path(__file__).parents[1]))
from src.data import RSMap
from src.ndp import NDP
from src.utils import build_scheduler
from src.radiomap.train import evaluate, visualize_history
from src.radiomap.visualize_rsrp import visualize_rsrp_map


def load_data(train_csv_paths, test_csv_path, batch_size):
    """
    加载多个训练集CSV文件和一个测试集CSV文件

    Args:
        train_csv_paths: 训练集CSV文件路径列表
        test_csv_path: 测试集CSV文件路径
        batch_size: 批次大小

    Returns:
        训练集DataLoader和测试集DataLoader
    """
    train_ds = ConcatDataset([RSMap(Path(csv_path)) for csv_path in train_csv_paths])
    print(f"合并后训练集样本总数: {len(train_ds)}")
    test_ds = RSMap(Path(test_csv_path))
    print(f"加载测试集: {test_csv_path}, 样本数: {len(test_ds)}")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_dl, test_dl


def train_transfer(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_dl, te_dl = load_data(cfg["train_csv_paths"], cfg["test_csv_path"], cfg["batch"])

    cfg["iter_per_epoch"] = len(tr_dl)

    save_dir = Path(cfg.get("save_dir", "results/transfer"))
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ndp_wrap = NDP(
        cfg["D"], cfg["T"], device, hidden=cfg["hidden"], n_layers=cfg["layers"]
    )
    model = ndp_wrap.model
    if "load_ckpt" in cfg and cfg["load_ckpt"]:
        model.load_state_dict(torch.load(cfg["load_ckpt"], map_location=device))

    opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-2)
    lr_sched = build_scheduler(opt, cfg)

    train_losses, val_rmses, val_pcrrs = [], [], []
    best_rmse = float("inf")
    for epoch in range(cfg["epochs"]):
        avg_loss = ndp_wrap.train_epoch(tr_dl, epoch, opt, device, lr_sched)
        train_losses.append(avg_loss)
        print(f"[Train] Epoch {epoch} | Avg Loss={avg_loss:.4f}")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            rmse_val, pcrr_val = evaluate(ndp_wrap, te_dl, device)
            val_rmses.append(rmse_val)
            val_pcrrs.append(pcrr_val)
            print(f"\033[32m[Val]   RMSE={rmse_val:.3f} | PCRR={pcrr_val:.3f}\033[0m")
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                torch.save(model.state_dict(), ckpt_dir / "ndp_best.pt")
    visualize_history(train_losses, val_rmses, val_pcrrs, save_dir)

    with open(save_dir / "transfer_results.txt", "w") as f:
        f.write(f"迁移学习结果\n")
        f.write(f"最佳测试 RMSE: {best_rmse:.6f}\n")
        f.write(f"最终测试 RMSE: {val_rmses[-1]:.6f}\n")
        f.write(f"最终测试 PCRR: {val_pcrrs[-1]:.6f}\n")
        f.write("\n训练配置:\n")
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")
    print(f"✓ 训练完成，结果已保存到 {save_dir}")
    visualize_rsrp_map(
        cfg,
        title="Transfer Test RSRP Map",
        resolution=256,
        full_coverage=True,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="迁移学习训练脚本")
    parser.add_argument("--cfg", default="configs/transfer.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    train_transfer(cfg)
