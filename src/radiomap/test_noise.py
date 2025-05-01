# scripts/test_noise.py
"""
NDP模型鲁棒性测试脚本 - 测试不同噪声水平下的模型性能
"""
import matplotlib.pyplot as plt
import torch
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm

from src.ndp import NDP
from src.radiomap.metrics import rmse, pcrr
from src.data import RSMap
from torch.utils.data import DataLoader


def load_model(cfg_path: str, ckpt_path: str, device: str = "cpu"):
    """加载预训练模型"""
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
    return model_wrap, cfg


def evaluate_with_noise(model, test_loader, device, noise_level=0.0):
    """在测试数据上评估模型在指定噪声水平下的性能"""
    y_true_all, y_pred_all = [], []
    model.model.eval()

    with torch.no_grad():
        for x, y in tqdm(
            test_loader, desc=f"[噪声水平={noise_level:.3f}]", leave=False
        ):
            x, y = x.to(device), y.to(device)

            if noise_level > 0:
                # 添加高斯噪声到特征
                noise = torch.randn_like(x) * noise_level
                x_noisy = x + noise
                # 确保噪声后的值仍在合理范围内 [0, 1]
                x_noisy = torch.clamp(x_noisy, 0, 1)
                y_hat = model.sample(x_noisy, 1)
            else:
                y_hat = model.sample(x, 1)

            y_true_all.append(y)
            y_pred_all.append(y_hat)

    y_true = torch.cat(y_true_all)
    y_pred = torch.cat(y_pred_all)

    return {
        "rmse": rmse(y_true, y_pred),
        "pcrr": pcrr(y_true, y_pred),
        "noise_level": noise_level,
    }


def visualize_performance_vs_noise(results, save_path=None):
    """绘制性能与噪声水平的关系曲线"""
    noise_levels = [r["noise_level"] for r in results]
    rmse_values = [r["rmse"] for r in results]
    pcrr_values = [r["pcrr"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # RMSE曲线 (越低越好)
    ax1.plot(
        noise_levels, rmse_values, "o-", color="#1f77b4", linewidth=2, markersize=8
    )
    ax1.set_xlabel("Noise Level (standard deviation)")
    ax1.set_ylabel("RMSE")
    ax1.set_title("Noise Level vs RMSE")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # 为每个点添加标签
    for i, (x, y) in enumerate(zip(noise_levels, rmse_values)):
        ax1.annotate(
            f"{y:.4f}", (x, y), xytext=(0, 5), textcoords="offset points", ha="center"
        )

    # PCRR曲线 (越高越好)
    ax2.plot(
        noise_levels, pcrr_values, "o-", color="#ff7f0e", linewidth=2, markersize=8
    )
    ax2.set_xlabel("Noise Level (standard deviation)")
    ax2.set_ylabel("PCRR")
    ax2.set_title("Noise Level vs PCRR")
    ax2.grid(True, linestyle="--", alpha=0.7)

    # 为每个点添加标签
    for i, (x, y) in enumerate(zip(noise_levels, pcrr_values)):
        ax2.annotate(
            f"{y:.4f}", (x, y), xytext=(0, 5), textcoords="offset points", ha="center"
        )

    plt.suptitle("NDP vs Noise Level", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"✅ 性能曲线已保存至: {save_path}")
    else:
        plt.show()

    # 创建文本报告
    if save_path:
        txt_path = Path(save_path).with_suffix(".txt")
        with open(txt_path, "w") as f:
            f.write("NDP模型噪声鲁棒性测试报告\n")
            f.write("=" * 50 + "\n\n")
            f.write("| 噪声水平 | RMSE    | PCRR    |\n")
            f.write("|----------|---------|--------|\n")
            for r in results:
                f.write(
                    f"| {r['noise_level']:.3f}    | {r['rmse']:.5f} | {r['pcrr']:.5f} |\n"
                )
        print(f"✅ 测试报告已保存至: {txt_path}")

    return fig


def run_robustness_test(args):
    """运行鲁棒性测试"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")

    # 加载模型
    print(f"📂 加载模型配置: {args.cfg}")
    print(f"📂 加载模型权重: {args.ckpt}")
    model, cfg = load_model(args.cfg, args.ckpt, device)

    # 加载测试数据集
    print(f"📊 加载测试数据: {args.csv}")
    dataset = RSMap(Path(args.csv))
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 运行不同噪声水平下的测试
    results = []
    for noise_level in args.noise_levels:
        print(f"🔍 测试噪声水平: {noise_level:.3f}")
        result = evaluate_with_noise(model, test_loader, device, noise_level)
        results.append(result)
        print(
            f"[噪声={noise_level:.3f}] RMSE={result['rmse']:.5f} | PCRR={result['pcrr']:.5f}"
        )

    # 可视化并保存结果
    save_path = None
    if args.save:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"noise_robustness_{Path(args.csv).stem}.png"

    visualize_performance_vs_noise(results, save_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试NDP模型在不同噪声水平下的鲁棒性")
    parser.add_argument(
        "--csv", default="datasets/processed/train_2304601.csv", help="测试数据CSV路径"
    )
    parser.add_argument("--cfg", default="configs/base.yaml", help="模型配置文件路径")
    parser.add_argument(
        "--ckpt", default="results/checkpoints/ndp_best.pt", help="模型权重文件路径"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="测试批次大小")
    parser.add_argument("--save", default="results/robustness", help="结果保存目录")
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
        help="要测试的噪声水平列表",
    )

    args = parser.parse_args()
    run_robustness_test(args)
