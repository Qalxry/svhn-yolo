import os
import cv2
import numpy as np
import shutil
import random
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(BASE_DIR, "tcdata")
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")

# 原始YOLO数据集目录
ORIGINAL_DATASET_DIR = os.path.join(DATA_DIR, "yolo_dataset")

# 增强后的数据集目录
ENHANCED_DATASET_DIR = os.path.join(DATA_DIR, "yolo_dataset_enhanced")

# 配置选项
USE_VAL_FOR_TRAIN = True  # 是否将验证集的一部分加入训练集
VAL_TRAIN_RATIO = 0.8  # 验证集中用于训练的比例
SAVE_AUGMENTATION_PREVIEW = True  # 是否预览增强结果
SHOW_PREVIEW = True  # 是否显示增强预览


def get_augmentation(extra=False):
    """
    创建数据增强管道

    Args:
        extra: 是否应用更强的增强

    Returns:
        配置好的增强管道
    """
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0 if extra else 0.7),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-15, 15),
                shear=(-10, 10),
                p=1.0 if extra else 0.7,
            ),
            A.Blur(blur_limit=3, p=0.5 if extra else 0.3),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.7 if extra else 0.5),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3, p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.5),
                ],
                p=0.3 if extra else 0.1,
            ),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                ],
                p=0.3 if extra else 0.1,
            ),
        ]
    )


# # 图像增强配置
# def get_augmentation(extra=False):
#     return A.Compose(
#         [
#             A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0 if extra else 0.5),
#             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0 if extra else 0.5),
#             A.Rotate(limit=25, p=1.0 if extra else 0.5),
#             A.Blur(blur_limit=3, p=1.0 if extra else 0.5),
#         ]
#     )


def read_yolo_label(label_path):
    """
    读取YOLO格式的标签文件

    Args:
        label_path: 标签文件路径

    Returns:
        两个列表：边界框和类别标签
    """
    bboxes = []
    class_labels = []

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)

    return bboxes, class_labels


def write_yolo_label(label_path, bboxes, class_labels):
    """
    写入YOLO格式的标签文件

    Args:
        label_path: 标签文件路径
        bboxes: 边界框列表
        class_labels: 类别标签列表
    """
    with open(label_path, "w") as f:
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            class_id = class_labels[i]
            line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
            f.write(line)


def move_partial_val_to_train(val_img_dir, val_label_dir, train_img_dir, train_label_dir, ratio=0.8):
    """
    将一部分验证集添加到训练集

    Args:
        val_img_dir: 验证集图像目录
        val_label_dir: 验证集标签目录
        train_img_dir: 训练集图像目录
        train_label_dir: 训练集标签目录
        ratio: 要添加到训练集的验证集比例
    """
    # 获取验证集图像列表
    val_img_files = [f for f in os.listdir(val_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

    # 随机选择比例的验证集
    random.shuffle(val_img_files)
    selected_count = int(len(val_img_files) * ratio)
    selected_files = val_img_files[:selected_count]

    print(f"将 {len(selected_files)} 个验证集图像 (总共 {len(val_img_files)}) 移动到训练集")

    # 添加到训练集，处理文件名冲突
    for img_file in tqdm(selected_files, desc="将验证集移动到训练集"):
        # 处理文件名冲突 - 添加前缀
        new_img_file = f"val_{img_file}"
        label_file = os.path.splitext(img_file)[0] + ".txt"
        new_label_file = f"val_{label_file}"

        # 移动图像和标签
        shutil.move(os.path.join(val_img_dir, img_file), os.path.join(train_img_dir, new_img_file))

        if os.path.exists(os.path.join(val_label_dir, label_file)):
            shutil.move(os.path.join(val_label_dir, label_file), os.path.join(train_label_dir, new_label_file))


def print_dataset_info(dataset_dir):
    """
    打印数据集信息

    Args:
        dataset_dir: 数据集目录
    """
    train_img_dir = os.path.join(dataset_dir, "train", "images")
    train_label_dir = os.path.join(dataset_dir, "train", "labels")
    val_img_dir = os.path.join(dataset_dir, "val", "images")
    val_label_dir = os.path.join(dataset_dir, "val", "labels")
    test_img_dir = os.path.join(dataset_dir, "test", "images")

    # 统计图像数量
    train_count = len([f for f in os.listdir(train_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))])
    val_count = len([f for f in os.listdir(val_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))])
    test_count = len([f for f in os.listdir(test_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))])
    train_label_count = len([f for f in os.listdir(train_label_dir) if f.endswith(".txt")])
    val_label_count = len([f for f in os.listdir(val_label_dir) if f.endswith(".txt")])
    test_label_count = len([f for f in os.listdir(test_img_dir) if f.endswith(".txt")])

    # 统计每个类别的数量
    class_counts = {i: 0 for i in range(10)}

    # 统计训练集类别
    for label_file in os.listdir(train_label_dir):
        label_path = os.path.join(train_label_dir, label_file)
        _, class_labels = read_yolo_label(label_path)
        for class_id in class_labels:
            class_counts[class_id] += 1

    print("\n数据集信息:")
    print(f"训练集图像数量: {train_count} (标签数量: {train_label_count})")
    print(f"验证集图像数量: {val_count} (标签数量: {val_label_count})")
    print(f"测试集图像数量: {test_count} (标签数量: {test_label_count})")
    print("\n训练集中各数字的出现次数:")
    for class_id, count in class_counts.items():
        print(f"数字 {class_id}: {count} 次")


def preview_augmentation_batch(img_dir, num_images=5, index_start=0, show=True):
    """
    预览增强结果

    Args:
        img_dir: 图像目录
        num_images: 要预览的图像数量
    """
    # 获取原始图像和增强图像
    img_files = [f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

    # 筛选出原始图像和它们的增强版本
    original_files = [f for f in img_files if not ("_aug" in f or "val_" in f)][index_start : num_images + index_start]

    if not original_files:
        print("没有找到适合预览的图像")
        return

    # 创建图形
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 2 * num_images))

    for i, orig_file in enumerate(original_files):
        # 读取原始图像
        orig_img = cv2.imread(os.path.join(img_dir, orig_file))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        # 查找此图像的增强版本
        base_name = os.path.splitext(orig_file)[0]
        aug1_file = f"{base_name}_aug1{os.path.splitext(orig_file)[1]}"
        aug2_file = f"{base_name}_aug2{os.path.splitext(orig_file)[1]}"

        # 读取增强图像
        if os.path.exists(os.path.join(img_dir, aug1_file)):
            aug1_img = cv2.imread(os.path.join(img_dir, aug1_file))
            aug1_img = cv2.cvtColor(aug1_img, cv2.COLOR_BGR2RGB)
        else:
            aug1_img = np.zeros_like(orig_img)

        if os.path.exists(os.path.join(img_dir, aug2_file)):
            aug2_img = cv2.imread(os.path.join(img_dir, aug2_file))
            aug2_img = cv2.cvtColor(aug2_img, cv2.COLOR_BGR2RGB)
        else:
            aug2_img = np.zeros_like(orig_img)

        # 显示图像
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Original: {orig_file}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(aug1_img)
        axes[i, 1].set_title(f"Augmented 1: {aug1_file}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(aug2_img)
        axes[i, 2].set_title(f"Augmented 2: {aug2_file}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    # 确保目录存在
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    # 保存预览图像
    print(f"保存增强预览图像到 {USER_DATA_DIR}")
    plt.savefig(
        os.path.join(USER_DATA_DIR, f"augmentation_preview_{index_start}-{index_start+num_images}.png"),
        dpi=300,
    )
    if show:
        plt.show()


def preview_augmentation(img_dir, num_images=6, batch_size=3, index_start=0, show=False):
    """
    预览增强结果

    Args:
        img_dir: 图像目录
        num_images: 要预览的图像数量
    """
    for i in range(index_start, num_images, batch_size):
        preview_augmentation_batch(img_dir, num_images=batch_size, index_start=i, show=show)


def augment_dataset(use_val_for_train=USE_VAL_FOR_TRAIN, preview=SAVE_AUGMENTATION_PREVIEW):
    """
    增强数据集并保存到新目录

    Args:
        use_val_for_train: 是否将验证集的一部分加入训练集
        preview: 是否预览增强结果
    """
    # 创建增强数据集的目录结构
    train_img_dir = os.path.join(ENHANCED_DATASET_DIR, "train", "images")
    train_label_dir = os.path.join(ENHANCED_DATASET_DIR, "train", "labels")
    val_img_dir = os.path.join(ENHANCED_DATASET_DIR, "val", "images")
    val_label_dir = os.path.join(ENHANCED_DATASET_DIR, "val", "labels")
    test_img_dir = os.path.join(ENHANCED_DATASET_DIR, "test", "images")

    # 创建目录
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    # 复制验证集和测试集（不增强）
    print("复制验证集和测试集...")
    copy_dataset(
        os.path.join(ORIGINAL_DATASET_DIR, "val", "images"),
        os.path.join(ORIGINAL_DATASET_DIR, "val", "labels"),
        val_img_dir,
        val_label_dir,
    )

    # 复制测试集图像
    test_image_list = [
        f
        for f in os.listdir(os.path.join(ORIGINAL_DATASET_DIR, "test", "images"))
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    for img_file in tqdm(test_image_list, desc="复制测试集图像"):
        src_path = os.path.join(ORIGINAL_DATASET_DIR, "test", "images", img_file)
        dst_path = os.path.join(test_img_dir, img_file)
        shutil.copy2(src_path, dst_path)

    # 首先复制原始训练集
    print("复制原始训练集...")
    copy_dataset(
        os.path.join(ORIGINAL_DATASET_DIR, "train", "images"),
        os.path.join(ORIGINAL_DATASET_DIR, "train", "labels"),
        train_img_dir,
        train_label_dir,
    )

    # 如果启用了将验证集加入训练集的功能
    if use_val_for_train:
        move_partial_val_to_train(
            val_img_dir,
            val_label_dir,
            train_img_dir,
            train_label_dir,
            ratio=VAL_TRAIN_RATIO,
        )

    # 增强训练集
    print("开始增强训练集...")
    augment_train_set(
        train_img_dir,
        train_label_dir,
        train_img_dir,
        train_label_dir,
    )

    # 创建数据集配置文件
    create_yolo_config(ENHANCED_DATASET_DIR)

    # 打印数据集信息
    print_dataset_info(ENHANCED_DATASET_DIR)

    # 预览增强结果
    if preview:
        preview_augmentation(train_img_dir, num_images=9, show=SHOW_PREVIEW)

    print(f"数据集增强完成，增强后的数据集保存在 {ENHANCED_DATASET_DIR}")


def copy_dataset(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir):
    """
    复制数据集从源目录到目标目录
    """
    src_image_list = [f for f in os.listdir(src_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

    # 复制图像
    for img_file in tqdm(src_image_list, desc="Copying images"):
        src_path = os.path.join(src_img_dir, img_file)
        dst_path = os.path.join(dst_img_dir, img_file)
        shutil.copy2(src_path, dst_path)

    # 复制标签（如果存在）
    if os.path.exists(src_label_dir):
        src_label_list = [f for f in os.listdir(src_label_dir) if f.endswith(".txt")]
        for label_file in tqdm(src_label_list, desc="Copying labels"):
            if not label_file.endswith(".txt"):
                continue
            # 确保标签文件与图像文件匹配
            src_path = os.path.join(src_label_dir, label_file)
            dst_path = os.path.join(dst_label_dir, label_file)
            shutil.copy2(src_path, dst_path)


def augment_train_set(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir):
    """
    对训练集进行数据增强

    Args:
        src_img_dir: 源图像目录
        src_label_dir: 源标签目录
        dst_img_dir: 目标图像目录
        dst_label_dir: 目标标签目录
    """
    # 获取所有图像文件
    img_files = [f for f in os.listdir(src_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

    # 统计稀有数字的分布
    class_counts = {i: 0 for i in range(10)}

    # 首先统计每个类别的数量
    for img_file in img_files:
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(src_label_dir, label_file)
        if os.path.exists(label_path):
            _, class_labels = read_yolo_label(label_path)
            for class_id in class_labels:
                class_counts[class_id] += 1

    print("原始数据集中各数字的出现次数:")
    for class_id, count in class_counts.items():
        print(f"数字 {class_id}: {count} 次")

    # 确定哪些类别较少，需要额外增强
    avg_count = sum(class_counts.values()) / len(class_counts)
    rare_classes = [class_id for class_id, count in class_counts.items() if count < avg_count * 0.7]

    print(f"较少出现的数字 (将获得额外增强): {rare_classes}")

    # 对每个图像进行增强
    for img_file in tqdm(img_files, desc="增强训练图像"):
        img_path = os.path.join(src_img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(src_label_dir, label_file)

        # 读取图像和标签
        image = cv2.imread(img_path)
        if image is None:
            continue

        bboxes, class_labels = read_yolo_label(label_path)

        # 修复：将bboxes转为numpy数组
        if len(bboxes) > 0:
            bboxes = np.array(bboxes, dtype=np.float32)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)

        # 检查是否包含稀有类别
        contains_rare = any(class_id in rare_classes for class_id in class_labels)

        # 确定增强次数，包含稀有类别的图像获得更多增强
        num_augmentations = 3 if contains_rare else 2

        # 应用增强
        for i in range(num_augmentations):
            # 使用不同的增强策略
            augmentation = get_augmentation(extra=contains_rare)

            # 应用增强
            augmented = augmentation(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_class_labels = augmented["class_labels"]

            # 生成新的文件名
            new_img_file = f"{os.path.splitext(img_file)[0]}_aug{i+1}{os.path.splitext(img_file)[1]}"
            new_label_file = f"{os.path.splitext(label_file)[0]}_aug{i+1}.txt"

            # 保存增强后的图像和标签
            cv2.imwrite(os.path.join(dst_img_dir, new_img_file), aug_image)
            write_yolo_label(os.path.join(dst_label_dir, new_label_file), aug_bboxes, aug_class_labels)


def create_yolo_config(yolo_dataset_dir):
    """
    创建YOLO数据集配置文件

    Args:
        yolo_dataset_dir: 数据集目录
    """
    config_content = f"""
path: {os.path.abspath(yolo_dataset_dir)}
train: train/images
val: val/images
test: test/images

nc: 10
names:
- 0
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
"""
    config_path = os.path.join(yolo_dataset_dir, "yolo_svhn.yaml")
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"YOLO配置文件已创建: {config_path}")


def completed_tag():
    """
    标记任务已完成: 在用户数据目录下创建一个标记文件
    """
    tag_file_path = os.path.join(USER_DATA_DIR, "data_augmented.txt")
    if os.path.exists(tag_file_path):
        print(f"数据增强任务已完成标记文件已存在: {tag_file_path}")
        return
    with open(tag_file_path, "w") as f:
        f.write("done")
    print(f"数据增强任务已完成标记文件已创建: {tag_file_path}")


if __name__ == "__main__":
    if not os.path.exists(os.path.join(USER_DATA_DIR, "data_augmented.txt")):
        print("开始进行图像数据集增强...")
        if os.path.exists(ENHANCED_DATASET_DIR):
            print(f"已删除旧的增强数据集目录: {ENHANCED_DATASET_DIR}")
            shutil.rmtree(ENHANCED_DATASET_DIR)
        os.makedirs(ENHANCED_DATASET_DIR, exist_ok=True)
        # 执行数据增强
        augment_dataset(use_val_for_train=USE_VAL_FOR_TRAIN, preview=SAVE_AUGMENTATION_PREVIEW)
        completed_tag()
    else:
        print("数据已增强，跳过。")
        if SAVE_AUGMENTATION_PREVIEW:
            print("预览增强结果...")
            preview_augmentation(os.path.join(ENHANCED_DATASET_DIR, "train", "images"), 9, show=SHOW_PREVIEW)
