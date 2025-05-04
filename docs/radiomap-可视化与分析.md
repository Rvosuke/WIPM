# 🎨 可视化与分析

## 🗺️ 栅格地图热力图可视化

使用脚本 `src/radiomap/visualize_rsrp.py`，可以将指定数据集上的 RSRP 真实值与预训练模型的预测值进行对比，生成并排的热力图。

### ⚙️ 工作流程

1.  **加载配置**: 脚本首先加载配置文件（默认为 `configs/base.yaml`，可通过 `--cfg` 参数指定）。
2.  **加载数据**: 根据配置文件中的 `csv` 路径加载包含 'X', 'Y', 'RSRP' 列的数据集。
3.  **加载模型**: 加载配置文件中 `ckpt` 指定的预训练 NDP 模型权重。
4.  **数据准备**:
    *   如果启用 `--full` 参数（默认启用），脚本会根据 `--resolution` 参数生成一个指定分辨率的完整坐标网格，用于全覆盖预测。
    *   如果禁用 `--full` 参数 (`--full False`)，脚本将仅使用原始 CSV 文件中的坐标点进行预测。
5.  **执行预测**: 使用加载的模型对准备好的坐标点（完整网格或原始数据点）进行 RSRP 预测。预测过程会分批次（batch）进行以优化内存使用。
6.  **生成热力图**:
    *   **左图 (Ground Truth)**: 使用原始数据中的 'X', 'Y', 'RSRP' 值绘制真实 RSRP 分布热力图。
    *   **右图 (Model Prediction)**: 使用模型预测的 RSRP 值绘制预测 RSRP 分布热力图。
7.  **保存输出**:
    *   将生成的热力图保存为 PNG 文件，路径由配置文件中的 `save` 参数指定。
    *   如果启用了 `--full` 参数，会将包含完整网格坐标及其预测 RSRP 值的数据保存为同名的 CSV 文件（例如，如果保存图像为 `results/map.png`，则 CSV 文件为 `results/map.csv`）。

### ✅ 支持功能

*   **Ground Truth 热力图**: 展示原始数据中的 RSRP 空间分布。
*   **Model Prediction 热力图**: 展示模型预测的 RSRP 空间分布。
*   **全覆盖预测 (可选)**: 可生成指定分辨率的完整网格预测图，更直观地展示模型在整个区域的预测效果。
*   **自定义配置**: 通过 YAML 文件和命令行参数灵活配置输入数据、模型、输出路径、图像标题和分辨率。
*   **结果保存**: 保存可视化图像及（可选的）全覆盖预测数据。

### 🛠️ 配置说明

可视化脚本主要通过配置文件和命令行参数进行设置。

**1. 配置文件 (`configs/base.yaml` 或通过 `--cfg` 指定)**

以下是配置文件中与可视化直接相关的关键参数：

*   `csv`: (必需) 输入的 CSV 数据集文件路径。该文件必须包含 `X`, `Y`, `RSRP` 列，以及模型输入所需的其他特征列。坐标 `X`, `Y` 应已归一化到 `[0, 1]` 区间。
*   `ckpt`: (必需) 预训练的模型权重文件路径 (`.pt` 文件)。
*   `save`: (必需) 输出的可视化图像保存路径（`.png` 文件）。如果启用全覆盖预测，同目录下会生成同名的 `.csv` 文件。
*   `D`: (必需) 模型的输入维度，需要与 `csv` 文件中的特征列数量（除 RSRP 外）匹配。
*   `T`, `hidden`, `layers`: (必需) 模型的超参数，需要与加载的 `ckpt` 文件训练时使用的参数一致。
*   `batch`: (可选, 默认 256) 预测时使用的批处理大小，影响内存占用和预测速度。

**示例 `configs/base.yaml` 相关部分:**

```yaml
# filepath: /home/baizy25/programs/WIPM/configs/base.yaml
csv: datasets/radiomap/train_2304601.csv # 输入数据集
batch: 256
# ... 其他训练参数 ...
D: 2 # 输入特征维度 (例如 X, Y)
T: 25
hidden: 256
layers: 6
# ... 其他训练参数 ...
ckpt: results/checkpoints/ndp_best.pt # 模型权重
save: results/radiomap/train_2304601_ndp.png # 输出图像路径
seed: 2025
```

**2. 命令行参数**

可以通过命令行参数覆盖或补充配置：

*   `--cfg`: 指定要使用的配置文件路径。默认为 `configs/base.yaml`。
*   `--title`: 设置生成图像的顶部显示的总标题(并非文件名)。默认为 "RSRP Heatmap"。
*   `--resolution`: 当启用全覆盖预测时，设置生成网格的分辨率（例如，256 表示生成 256x256 的网格）。默认为 256。
*   `--full`: 是否执行全覆盖预测。这是一个标志参数，存在即为 True。默认为 True (启用)。

### 📌 命令示例

**1. 使用默认配置进行可视化:**

```bash
python -m src.radiomap.visualize_rsrp
```
这将使用 `configs/base.yaml` 中的设置，生成 256x256 分辨率的全覆盖预测图，标题为 "RSRP Heatmap"，并将结果保存到 `results/radiomap/train_2304601_ndp.png`。

**2. 指定配置文件、自定义标题和分辨率:**

```bash
python -m src.radiomap.visualize_rsrp \
    --cfg configs/another_model.yaml \
    --title "Site B RSRP Prediction (512x512)" \
    --resolution 512
```
这将使用 `configs/another_model.yaml` 进行配置，生成 512x512 分辨率的全覆盖预测图，并设置自定义标题。输出路径由 `another_model.yaml` 中的 `save` 参数决定。


### 🎨 色彩规范

*   所有 `RSRP` 热力图（真实值和预测值）的归一化色条范围统一设置为 `[0, 1]`，使用 `YlGnBu` 色彩映射，方便对比。

### 💾 输出说明

*   **PNG 图像**: 生成的图像文件包含两个并排的热力图。
    *   左侧：**Ground Truth**，基于输入 CSV 中的实际 RSRP 数据。
    *   右侧：**Model Prediction**，基于模型对相应坐标（全覆盖网格或原始点）的预测 RSRP 值。
*   **CSV 文件 (可选)**: 当启用 `--full` (默认) 时，会额外生成一个 CSV 文件。该文件包含 `--resolution` 指定的完整网格的 `X`, `Y` 坐标以及模型对每个点的预测 RSRP 值 (`RSRP_PRED`)。这对于后续更详细的定量分析非常有用。
