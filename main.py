"""
简单入口：
    python main.py train   # 训练
    python main.py sample # 生成整张 RSRP 热力图 (待实现)
"""

from __future__ import annotations
import argparse, sys, yaml
from pathlib import Path


def load_cfg(yaml_path: str = "config.yaml"):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["train", "sample"])
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.cmd == "train":
        from src.train import train

        train(cfg)
    elif args.cmd == "sample":
        # TODO: 实现条件 / 无条件采样并存储为 csv/heatmap
        print("Sampling not yet implemented.")
    else:
        sys.exit("Unknown command")


if __name__ == "__main__":
    main()
