import os
import glob
import torch
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(BASE_DIR, "tcdata")
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")
RESULT_DIR = os.path.join(BASE_DIR, "prediction_result")
YOLO_DATASET_DIR = os.path.join(DATA_DIR, "yolo_dataset")  # 只是取用yolo_dataset目录下的test/images
os.makedirs(RESULT_DIR, exist_ok=True)

# 推理参数配置
# Number of images to predict in each batch
BATCH_SIZE = 196
# Confidence threshold for predictions
# Extremely important. If set too high, some digits may be missed and accuracy will decrease 5-8%
CONF_THRESHOLD = 0.3


def predict_and_save_results():
    # 创建结果目录
    os.makedirs("prediction_result", exist_ok=True)

    # 加载最佳模型
    # 根据需要修改模型路径和输出名称
    output_name = "best"
    model_path = os.path.join(USER_DATA_DIR, "model_data/best.pt")
    
    # # Trained on original train set
    # output_name = "original_train_set"
    # model_path = os.path.join(USER_DATA_DIR, "remote/new/user_data/train_20250529_012023/weights/best.pt")

    # # Trained on data which added 8k validation images to the train set
    # output_name = "train_set_with_8k_val"
    # model_path = os.path.join(USER_DATA_DIR, "remote/new/user_data/train_20250529_012931/weights/best.pt")

    # # Trained on augmented data which added 8k validation images to the train set
    # output_name = "train_set_with_8k_val_augmented"
    # model_path = os.path.join(USER_DATA_DIR, "remote/new/user_data/train_20250529_003910/weights/best.pt")

    model = YOLO(model_path)

    # 获取测试图像路径
    test_images_dir = os.path.join(YOLO_DATASET_DIR, "test", "images")
    image_paths: list = glob.glob(os.path.join(test_images_dir, "*.png"))

    # sort
    image_paths.sort()

    results = []

    # 批量处理图像
    batch_size = BATCH_SIZE
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = [Image.open(img_path) for img_path in batch_paths]

        # 批量预测
        preds = model(
            batch_images,
            imgsz=320,
            conf=CONF_THRESHOLD,
            batch=batch_size,
            verbose=False,
            device="0" if torch.cuda.is_available() else "cpu",
        )

        # 处理预测结果
        for j, (img_path, pred) in enumerate(zip(batch_paths, preds)):
            digits = []
            for box in pred.boxes:
                # 获取数字类别和位置
                digit = int(box.cls)
                x_center = box.xywh[0][0].item()

                # 保存数字和位置
                digits.append((x_center, digit))

            # 按x坐标排序数字
            digits.sort()
            file_code = "".join([str(d[1]) for d in digits])

            # 获取文件名
            file_name = os.path.basename(img_path)

            results.append({"file_name": file_name, "file_code": file_code})

    # 保存为CSV
    df = pd.DataFrame(results)
    output_path = os.path.join(RESULT_DIR, f"result_{output_name}.csv")
    df.to_csv(output_path, index=False)
    print(f"预测结果已保存到: '{output_path}'")


if __name__ == "__main__":
    predict_and_save_results()
