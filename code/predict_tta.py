import os
import glob
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
from collections import Counter

from augment import get_augmentation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(BASE_DIR, "tcdata")
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")
RESULT_DIR = os.path.join(BASE_DIR, "prediction_result")
YOLO_DATASET_DIR = os.path.join(DATA_DIR, "yolo_dataset")
os.makedirs(RESULT_DIR, exist_ok=True)

# 加载最佳模型，根据需要修改模型路径和输出名称
OUTPUT_NAME = "best"
MODEL_PATH = os.path.join(USER_DATA_DIR, "model_data/best.pt")
# OUTPUT_NAME = "train_set_with_8k_val_augmented_final_tta"
# MODEL_PATH = os.path.join(USER_DATA_DIR, "remote/final/best.pt")

# 推理参数
BATCH_SIZE = 196
CONF_THRESHOLD = 0.3

# TTA配置（Test Time Augmentation，测试时增强）
ENABLE_TTA = True
APPLY_TTA_FOR_HARD_CASES_ONLY = True  # 只对难例使用TTA
TTA_CONFIDENCE_THRESHOLD = 0.5  # 如果有检测置信度低于该阈值，则应用TTA
TTA_NUM_AUGMENTATIONS = 6  # 增强次数


def process_single_prediction(pred, img_path):
    """处理单张图片的预测结果，返回文件名和识别码"""
    digits = []
    confidence_scores = []

    for box in pred.boxes:
        # 获取数字类别和位置
        digit = int(box.cls)
        x_center = box.xywh[0][0].item()
        confidence = box.conf[0].item()

        # 保存数字、位置和置信度
        digits.append((x_center, digit))
        confidence_scores.append(confidence)

    # 按x坐标排序
    digits.sort()
    file_code = "".join([str(d[1]) for d in digits])

    # 获取文件名
    file_name = os.path.basename(img_path)

    needs_tta = False
    if ENABLE_TTA:
        if APPLY_TTA_FOR_HARD_CASES_ONLY:
            # 如果只对难例使用TTA，则检查是否有置信度低于阈值的检测
            needs_tta = any(conf < TTA_CONFIDENCE_THRESHOLD for conf in confidence_scores) if confidence_scores else True
        else:
            # 如果对所有图片都使用TTA，则直接设置需要TTA
            needs_tta = True
    
    return {
        "file_name": file_name,
        "file_code": file_code,
        "confidence_scores": confidence_scores,
        "needs_tta": needs_tta,
    }


def apply_tta(model, image_path):
    """对图片应用TTA并返回投票结果"""
    # 读取原始图片
    image = cv2.imread(image_path)
    if image is None:
        return

    # 转换为RGB格式
    augmented_images = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))]  # 原图PIL格式

    # 添加增强版本（已经有原图）
    for i in range(TTA_NUM_AUGMENTATIONS):
        # PIL转numpy，增强，再转回PIL
        augmented_img = get_augmentation()(image=image, bboxes=[], class_labels=[])["image"]
        augmented_pil = Image.fromarray(cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB))
        augmented_images.append(augmented_pil)

    # 对所有增强版本进行预测
    predictions = []

    # 分批处理，避免显存溢出
    preds = model(
        augmented_images,
        imgsz=320,
        conf=CONF_THRESHOLD,
        verbose=False,
        device="0" if torch.cuda.is_available() else "cpu",
    )

    for pred in preds:
        digits = []
        for box in pred.boxes:
            digit = int(box.cls)
            x_center = box.xywh[0][0].item()
            digits.append((x_center, digit))

        # 按x坐标排序
        digits.sort()
        file_code = "".join([str(d[1]) for d in digits])
        predictions.append(file_code)

    # 统计出现最多的预测结果
    if predictions:
        counter = Counter(predictions)
        consensus_prediction = counter.most_common(1)[0][0]
    else:
        # 没有预测结果则返回空字符串
        consensus_prediction = ""

    return consensus_prediction


def predict_and_save_results():
    # 创建结果目录
    os.makedirs("prediction_result", exist_ok=True)

    # 加载模型
    output_name = OUTPUT_NAME
    model_path = MODEL_PATH
    model = YOLO(model_path)

    # 获取测试图片路径
    test_images_dir = os.path.join(YOLO_DATASET_DIR, "test", "images")
    image_paths = glob.glob(os.path.join(test_images_dir, "*.png"))
    image_paths.sort()

    results = []
    tta_candidates = []

    # 第一轮：对所有图片常规推理
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="第一轮处理"):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        batch_images = [Image.open(img_path) for img_path in batch_paths]

        # 批量预测
        preds = model(
            batch_images,
            imgsz=320,
            conf=CONF_THRESHOLD,
            batch=BATCH_SIZE,
            verbose=False,
            device="0" if torch.cuda.is_available() else "cpu",
        )

        # 处理预测结果
        for j, (img_path, pred) in enumerate(zip(batch_paths, preds)):
            # 处理单张图片的预测结果：文件名、识别结果、置信度、是否需要TTA
            result = process_single_prediction(pred, img_path)

            if result["needs_tta"]:
                # 如果启用TTA且该样本为难样本需要TTA，则加入候选，后续处理
                tta_candidates.append(img_path)
            else:
                # 否则直接加入结果
                results.append({"file_name": result["file_name"], "file_code": result["file_code"]})

    print(f"无需TTA处理的图片数: {len(results)}")
    print(f"需要TTA的低置信度样本数: {len(tta_candidates)}")
    print(f"总图片数: {len(image_paths)}")

    # 保存TTA候选样本
    tta_candidates_path = os.path.join(USER_DATA_DIR, "temp_tta_candidates.txt")
    with open(tta_candidates_path, "w") as f:
        for img_path in tta_candidates:
            f.write(f"{img_path}\n")
    print(f"低置信度样本已保存到: '{tta_candidates_path}'")

    # 保存临时结果
    temp_results_path = os.path.join(USER_DATA_DIR, "temp_results.csv")
    temp_df = pd.DataFrame(results)
    temp_df.to_csv(temp_results_path, index=False)
    print(f"临时结果已保存到: '{temp_results_path}'")

    # tta_candidates = []
    # with open(os.path.join(RESULT_DIR, "temp_tta_candidates.txt"), "r") as f:
    #     for line in f:
    #         tta_candidates.append(line.strip())
    # print(f"从文件加载低置信度样本数: {len(tta_candidates)}")

    # 第二轮：对低置信度样本应用TTA
    if ENABLE_TTA and tta_candidates:
        for img_path in tqdm(tta_candidates, desc="应用TTA到低置信度样本"):
            consensus_prediction = apply_tta(model, img_path)
            results.append({"file_name": os.path.basename(img_path), "file_code": consensus_prediction})

    # 按文件名排序
    results.sort(key=lambda x: x["file_name"])

    # 保存最终结果到CSV
    df = pd.DataFrame(results)
    output_path = os.path.join(RESULT_DIR, f"result_{output_name}.csv")
    df.to_csv(output_path, index=False)
    print(f"预测结果已保存到: '{output_path}'")
    print(f"TTA应用于{len(tta_candidates)}个低置信度样本，共{len(image_paths)}张图片。")


if __name__ == "__main__":
    predict_and_save_results()
