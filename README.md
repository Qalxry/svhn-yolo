# SVHN 字符识别项目（基于 YOLO11）

本项目旨在使用 YOLO11 (You Only Look Once) 模型对街景门牌号（SVHN）数据集中的数字序列进行检测和识别。项目流程涵盖了数据自动下载、格式转换、数据增强、模型训练和结果预测。

## 项目特性

- **自动化流程**: 使用 Shell 脚本一键执行从数据准备到模型训练、再到结果预测的全过程。
- **数据处理**: 自动从天池竞赛源下载 SVHN 数据集，并将其从 JSON 格式转换为 YOLO11 所需的 TXT 格式。
- **高级数据增强**: 利用 `albumentations` 库对训练数据进行复杂的数据增强，包括仿射变换、亮度对比度调整、模糊等，并对样本较少的数字类别进行额外的增强，以提升模型的泛化能力。
- **模型训练**: 基于 `ultralytics` 框架，使用预训练的 YOLO11m 模型在增强后的 SVHN 数据集上进行微调。
- **结果生成**: 自动对测试集图像进行预测，将识别出的数字序列按从左到右的顺序排列，并生成符合竞赛提交要求的 `result.csv` 文件。

## 文件结构

```
.
├── README.md                  # 项目说明文档
├── tcdata
│   ├── raw_data/              # 存放下载的原始数据和解压后的文件 (运行后生成)
│   ├── yolo_dataset/          # 转换为YOLO格式的原始数据集 (运行后生成)
│   └── yolo_dataset_enhanced/ # 经过增强后的最终数据集 (运行后生成)
├── code
│   ├── get_dataset.py         # 下载并转换数据集脚本
│   ├── augment.py             # 数据增强脚本
│   ├── train.py               # 模型训练脚本
│   ├── predict.py             # 预测脚本
│   ├── test.sh                # 执行预测流程
│   └── train.sh               # 执行数据准备和训练流程
├── prediction_result          # 存放预测结果的目录 (运行后生成)
│   └── result.csv             # 最终的预测结果 (运行后生成)
└── user_data
     ├── model_data
     │   └── yolo_svhn_best.pt  # 训练好的最佳模型权重 (运行后生成)
     ├── yolo11n.pt             # 预训练的YOLO模型 (运行后生成)
     ├── data_downloaded.txt    # 数据下载完成的标记文件 (运行后生成)
     └── data_augmented.txt     # 数据增强完成的标记文件 (运行后生成)
```

## 环境准备

1. **克隆项目**:

    ```bash
    git clone https://github.com/Qalxry/svhn-yolo.git
    cd svhn-yolo
    ```

2. **创建 Conda 环境**:

    ```bash
    conda create -n yolo python=3.12
    conda activate yolo
    ```

3. **安装依赖**:
    项目依赖于多个 Python 库。您可以通过 `requirements.txt` 文件安装：

    ```bash
    pip install -r requirements.txt
    ```

## 使用说明

项目提供了两个主要的 Shell 脚本来简化操作流程：`train.sh` 用于数据准备和模型训练，`test.sh` 用于生成预测结果。

### 1. 训练模型

执行以下命令，将自动完成数据下载、格式转换、数据增强和模型训练的全过程：

```bash
bash code/train.sh
```

此脚本会依次执行：

1. `code/get_dataset.py`:
    - 从天池服务器下载原始的 SVHN 数据集压缩包和 JSON 标注文件。
    - 解压文件到 `tcdata/raw_data/`。
    - 将数据转换为 YOLO 格式，并存放在 `tcdata/yolo_dataset/`。
    - 创建 `user_data/data_downloaded.txt` 标记文件，防止重复执行。
2. `code/augment.py`:
    - 在 `tcdata/yolo_dataset_enhanced/` 中创建增强后的数据集。
    - **策略**: 将部分验证集数据并入训练集，并对训练集进行数据增强（对稀有数字进行更强的增强）。
    - 创建 `user_data/data_augmented.txt` 标记文件。
3. `code/train.py`:
    - 加载 `user_data/` 目录下的预训练模型。
    - 使用 `tcdata/yolo_dataset_enhanced/` 中的数据进行训练。
    - 训练完成后，最佳模型将保存为 `user_data/model_data/yolo_svhn_best.pt`。

### 2. 生成预测结果

模型训练完成后，执行以下命令来对测试集进行预测：

```bash
bash code/test.sh
```

此脚本会执行 `code/predict.py`：

- 加载训练好的最佳模型 `user_data/model_data/yolo_svhn_best.pt`。
- 对 `tcdata/yolo_dataset_enhanced/test/images/` 目录下的所有图片进行推理。
- 对于每张图片，脚本会识别出所有数字，并根据它们的横向位置（x 坐标）从左到右排序，拼接成一个字符串。
- 最终结果将保存到 `prediction_result/result.csv` 文件中，格式如下：

| file_name | file_code |
| :-------- | :-------- |
| 00000.png.txt | 19 |
| 00001.png.txt | 25 |
| ... | ... |

## 脚本详解

- **`code/get_dataset.py`**:
  - 负责从网络下载数据集。
  - 调用 `convert_to_yolo_format` 函数，将 JSON 格式的标签（包含每个数字的位置和值）转换为 YOLO 需要的 `.txt` 格式（`class_id x_center y_center width height`）。

- **`code/augment.py`**:
  - 核心功能是扩充和丰富训练数据，以防止过拟合。
  - 使用 `albumentations` 库定义了一系列增强操作，如随机亮度对比度、仿射变换（缩放、平移、旋转、剪切）、模糊、颜色通道偏移等。
  - 通过分析数据集中各数字的出现频率，对包含稀有数字的图像应用更频繁或更强的增强，以平衡类别分布。
  - 提供 `PREVIEW_AUGMENTATION` 开关，可以在 `user_data` 目录下生成增强效果的预览图。

- **`code/train.py`**:
  - 封装了 `ultralytics` 库的训练接口。
  - 训练配置（如 `epochs`, `imgsz`, `patience` 等）清晰地定义在 `train_config` 字典中，方便调整。
  - 训练过程会自动利用可用的 GPU。

- **`code/predict.py`**:
  - 加载训练好的模型。
  - 遍历测试图片，进行预测。
  - 关键处理步骤：对检测到的边界框按 `x` 坐标排序，以确保数字序列的正确性。
