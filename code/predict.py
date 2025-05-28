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
YOLO_DATASET_DIR = os.path.join(DATA_DIR, "yolo_dataset_enhanced")

os.makedirs(RESULT_DIR, exist_ok=True)


def predict_and_save_results():
    # 创建结果目录
    os.makedirs("prediction_result", exist_ok=True)

    # 加载最佳模型
    model_path = os.path.join(USER_DATA_DIR, "model_data", "yolo_svhn_best.pt")
    model = YOLO(model_path)

    # 获取测试图像路径
    test_images_dir = os.path.join(YOLO_DATASET_DIR, "test", "images")
    image_paths: list = glob.glob(os.path.join(test_images_dir, "*.png"))

    # sort
    image_paths.sort()

    results = []

    for img_path in tqdm(image_paths, desc="Processing images"):
        # 预测
        img = Image.open(img_path)
        preds = model(img, imgsz=320, conf=0.5, verbose=False, device="0" if torch.cuda.is_available() else "cpu")

        # 处理预测结果
        digits = []
        for box in preds[0].boxes:
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
    output_path = os.path.join(RESULT_DIR, "result.csv")
    df.to_csv(output_path, index=False)
    print(f"预测结果已保存到: '{output_path}'")


if __name__ == "__main__":
    predict_and_save_results()
