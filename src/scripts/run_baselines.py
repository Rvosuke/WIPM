# scripts/run_baselines.py
import argparse
import yaml
import sys
import os
from pathlib import Path

# 将父目录添加到路径，以便导入src模块
sys.path.append(str(Path(__file__).parent.parent))
from src.baseline_models import train_and_evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="运行基线模型(线性回归和SVR)并与NDP模型进行对比"
    )
    parser.add_argument("--cfg", default="configs/base.yaml", help="配置文件路径")
    parser.add_argument("--save_dir", default="results", help="结果保存目录")

    # 线性回归参数
    parser.add_argument(
        "--linear_fit_intercept", type=bool, default=True, help="线性回归是否包含截距项"
    )

    # SVR参数
    parser.add_argument(
        "--svr_kernel",
        type=str,
        default="rbf",
        choices=["linear", "poly", "rbf", "sigmoid"],
        help="SVR核函数",
    )
    parser.add_argument("--svr_C", type=float, default=1.0, help="SVR正则化参数")
    parser.add_argument(
        "--svr_epsilon", type=float, default=0.1, help="SVR epsilon参数"
    )
    parser.add_argument("--svr_gamma", type=str, default="scale", help="SVR gamma参数")

    args = parser.parse_args()

    # 加载配置
    cfg = yaml.safe_load(open(args.cfg))

    # 添加命令行参数到配置
    cfg["save_dir"] = args.save_dir

    # 添加线性回归参数
    cfg["linear_fit_intercept"] = args.linear_fit_intercept

    # 添加SVR参数
    cfg["svr_kernel"] = args.svr_kernel
    cfg["svr_C"] = args.svr_C
    cfg["svr_epsilon"] = args.svr_epsilon
    cfg["svr_gamma"] = args.svr_gamma

    print("=" * 50)
    print("📊 开始训练与评估基线模型...")
    print("=" * 50)

    results = train_and_evaluate(cfg)

    print("\n" + "=" * 50)
    print("✅ 评估完成！结果已保存至:", Path(cfg["save_dir"]))
    print("=" * 50)
