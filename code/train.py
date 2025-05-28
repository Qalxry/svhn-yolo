import os
import datetime
import torch
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(BASE_DIR, "tcdata")
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")
# YOLO_DATASET_DIR = os.path.join(DATA_DIR, "yolo_dataset")
YOLO_DATASET_DIR = os.path.join(DATA_DIR, "yolo_dataset_enhanced")

os.makedirs(USER_DATA_DIR, exist_ok=True)


def train_yolo():
    # 加载YOLO模型
    model = YOLO(os.path.join(USER_DATA_DIR, "yolo11m.pt"))  # 使用预训练模型
    
    # 训练配置
    train_config = {
        "project": USER_DATA_DIR,
        "name": f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",  # 项目名称
        "data": os.path.join(YOLO_DATASET_DIR, "yolo_svhn.yaml"),
        "epochs": 100,  # 训练周期
        # "batch": 32,  # 批量大小
        "imgsz": 320,  # 图片大小
        "device": "0" if torch.cuda.is_available() else "cpu",  # 使用GPU或CPU
        "verbose": True,  # 详细信息
        # "resume": True,  # 接着训练
        "save": True,  # 保存结果
        "patience": 10,  # 停止周期数
        "plots": True,  # 训练集与验证集性能
        # # 数据增强设置
        # "augment": True,
        # "hsv_h": 0.005,
        # "hsv_s": 0.7,
        # "hsv_v": 0.5,
        # "degrees": 25,
        # "translate": 0.15,
        # "scale": 0.5,
        # "shear": 0.05,
        # "perspective": 0.0,
        # "flipud": 0.0,
        # "fliplr": 0.0,
        # "bgr": 0.05,
        # "mosaic": 0.9,
        # "mixup": 0.01,
        # "copy_paste": 0.01,
        # "erasing": 0.1,
        # "crop_fraction": 0.5,
    }

    # 开始训练
    results = model.train(**train_config)

    # 创建模型保存目录
    os.makedirs(os.path.join(USER_DATA_DIR, "model_data"), exist_ok=True)
    # 保存最佳模型
    best_model_path = os.path.join(os.path.join(USER_DATA_DIR, "model_data"), "yolo_svhn_best.pt")
    model.save(best_model_path)
    print(f"训练完成，最佳模型已保存到: {best_model_path}")

    return model


if __name__ == "__main__":
    train_yolo()
