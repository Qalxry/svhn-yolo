import os
import cv2
import numpy as np
import shutil
import random
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt
import concurrent.futures

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(BASE_DIR, "tcdata")
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")

# 原始YOLO数据集目录
ORIGINAL_DATASET_DIR = os.path.join(DATA_DIR, "yolo_dataset")

# 增强后的数据集目录
ENHANCED_DATASET_DIR = os.path.join(DATA_DIR, "yolo_dataset_enhanced")

# 配置选项
NUM_WORKERS = os.cpu_count()  # 使用CPU核心数的4倍作为工作线程数
USE_VAL_FOR_TRAIN = True  # 是否将验证集的一部分加入训练集
VAL_TRAIN_RATIO = 0.8  # 验证集中用于训练的比例
SAVE_AUGMENTATION_PREVIEW = True  # 是否预览增强结果
SHOW_PREVIEW = True  # 是否显示增强预览
AUGMENT_COUNT = 4
RARE_MORE_AUGMENT_COUNT = 2

# # 图像增强配置 (Basic)
# def get_augmentation(extra=False, **kwargs):
#     return A.Compose(
#         [
#             A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0 if extra else 0.5),
#             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0 if extra else 0.5),
#             A.Rotate(limit=25, p=1.0 if extra else 0.5),
#             A.Blur(blur_limit=3, p=1.0 if extra else 0.5),
#         ],
#         bbox_params=A.BboxParams(
#             format="yolo",
#             label_fields=["class_labels"],
#             min_visibility=0.0,
#             clip=True,
#             filter_invalid_bboxes=True,
#         ),
#     )

# # 图像增强配置 (Advanced)
# def get_augmentation(extra=False):
#     """
#     创建数据增强管道

#     Args:
#         extra: 是否应用更强的增强

#     Returns:
#         配置好的增强管道
#     """
#     return A.Compose(
#         [
#             A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0 if extra else 0.7),
#             A.Affine(
#                 scale=(0.8, 1.2),
#                 translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
#                 rotate=(-15, 15),
#                 shear=(-10, 10),
#                 p=1.0 if extra else 0.7,
#             ),
#             A.Blur(blur_limit=3, p=0.5 if extra else 0.3),
#             A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.7 if extra else 0.5),
#             A.OneOf(
#                 [
#                     A.MotionBlur(blur_limit=3, p=0.5),
#                     A.MedianBlur(blur_limit=3, p=0.5),
#                     A.GaussianBlur(blur_limit=3, p=0.5),
#                 ],
#                 p=0.3 if extra else 0.1,
#             ),
#             A.OneOf(
#                 [
#                     A.CLAHE(clip_limit=2),
#                     A.Sharpen(),
#                     A.Emboss(),
#                 ],
#                 p=0.3 if extra else 0.1,
#             ),
#         ],
#         bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0),
#     )


# 图像增强配置 (Ultra SOTA)
def get_augmentation(extra=False, is_small=False):
    """
    超级 SOTA 图像增强管道

    Args:
        extra: 是否应用更强的增强

    Returns:
        A.Compose 增强流水线
    """
    return A.Compose(
        [
            # 亮度 / 对比度 / 色相 / 饱和度
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.4, p=1),
                    A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
                ],
                p=1.0 if extra else 0.8,
            ),
            # 几何变换 (使用仿射变换替换 ShiftScaleRotate 和 Perspective)
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-15, 15),
                shear=(-15, 15),
                fit_output=False,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0 if extra else 0.7,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=1,
                        sigma=50,
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_REPLICATE,
                        p=1,
                    ),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
                    A.OpticalDistortion(
                        distort_limit=(-0.2, 0.2),
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_CONSTANT,
                        p=1,
                    ),
                ],
                p=0.8 if extra else 0.5,
            ),
            # 噪声 / 压缩 / 下采样
            A.OneOf(
                [
                    A.GaussNoise(
                        std_range=(0.1, 0.3), mean_range=(0, 0), per_channel=True, noise_scale_factor=0.1, p=1
                    ),
                    A.ISONoise(color_shift=(0.01, 0.05), p=1),
                    A.ImageCompression(quality_range=(30, 100), p=1),
                    A.Downscale(scale_range=(0.5, 0.75), p=1),
                ],
                p=0.6 if extra and not is_small else 0.0,
            ),
            # 模糊
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3, p=1),
                    A.MedianBlur(blur_limit=3, p=1),
                    A.GaussianBlur(blur_limit=2, p=1),
                    # A.GlassBlur(sigma=0.4, max_delta=2, iterations=2, p=1),
                ],
                p=0.5 if extra and not is_small else 0.0,
            ),
            # 天气/光照效果
            A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.3 if extra else 0.1),

            # 颜色通道打乱 / 灰度
            A.OneOf(
                [
                    A.ChannelShuffle(p=1),
                    # A.ToGray(p=0.3),
                ],
                p=0.6 if extra else 0.3,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.0,
            clip=True,
            filter_invalid_bboxes=True,
        ),
    )


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
            line = f"{int(class_id)} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
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
    预览增强结果，并绘制检测框（使用matplotlib的patches和不同颜色/标签）

    Args:
        img_dir: 图像目录
        num_images: 要预览的图像数量
    """
    import matplotlib.patches as patches

    # 获取原始图像和增强图像
    img_files = [f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    # 筛选出原始图像和它们的增强版本
    original_files = [f for f in img_files if not ("_aug" in f or "val_" in f)][index_start : num_images + index_start]

    if not original_files:
        print("没有找到适合预览的图像")
        return

    # 颜色列表
    color_list = [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#bcf60c",
        "#fabebe",
    ]

    def draw_boxes_matplotlib(ax, image, label_path):
        img_h, img_w = image.shape[:2]
        ax.imshow(image)
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x, y, w, h = map(float, parts)
                        # YOLO格式转为像素坐标
                        x1 = (x - w / 2) * img_w
                        y1 = (y - h / 2) * img_h
                        box_w = w * img_w
                        box_h = h * img_h
                        color = color_list[int(class_id) % len(color_list)]
                        rect = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                        ax.add_patch(rect)
                        ax.text(
                            x1,
                            y1 - 2,
                            f"{int(class_id)}",
                            color=color,
                            fontsize=10,
                            weight="bold",
                            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=0),
                        )
        ax.axis("off")

    fig, axes = plt.subplots(num_images, AUGMENT_COUNT+1, figsize=(AUGMENT_COUNT*4, 3 * num_images))

    for i, orig_file in enumerate(original_files):
        # 读取原始图像
        orig_img = cv2.imread(os.path.join(img_dir, orig_file))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        orig_label = os.path.splitext(orig_file)[0] + ".txt"
        orig_label_path = os.path.join(os.path.dirname(img_dir), "labels", orig_label)

        # 查找此图像的增强版本
        base_name = os.path.splitext(orig_file)[0]

        # 原图
        ax0 = axes[i, 0] if num_images > 1 else axes[0]
        draw_boxes_matplotlib(ax0, orig_img, orig_label_path)
        ax0.set_title(f"Original: {orig_file}")

        for j in range(AUGMENT_COUNT):
            aug_file = f"{base_name}_aug{j+1}{os.path.splitext(orig_file)[1]}"
            aug_label = f"{base_name}_aug{j+1}.txt"
            aug_label_path = os.path.join(os.path.dirname(img_dir), "labels", aug_label)
            
            # 增强1
            if os.path.exists(os.path.join(img_dir, aug_file)):
                aug_img = cv2.imread(os.path.join(img_dir, aug_file))
                aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                draw_boxes_matplotlib(axes[i, j+1], aug_img, aug_label_path)
            else:
                axes[i, j+1].imshow(np.zeros_like(orig_img))
                axes[i, j+1].axis("off")
            axes[i, j+1].set_title(f"Augmented {j+1}: {aug_file}")

    plt.tight_layout()
    os.makedirs(USER_DATA_DIR, exist_ok=True)
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


def copy_image(args):
    src_img_dir, dst_img_dir, img_file = args
    src_path = os.path.join(src_img_dir, img_file)
    dst_path = os.path.join(dst_img_dir, img_file)
    shutil.copy2(src_path, dst_path)


def copy_label(args):
    src_label_dir, dst_label_dir, label_file = args
    src_path = os.path.join(src_label_dir, label_file)
    dst_path = os.path.join(dst_label_dir, label_file)
    shutil.copy2(src_path, dst_path)


def copy_dataset(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir):
    """
    复制数据集从源目录到目标目录，使用进程池加速
    """
    src_image_list = [f for f in os.listdir(src_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    img_args = [(src_img_dir, dst_img_dir, img_file) for img_file in src_image_list]

    # 复制图像（多进程加速）
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(copy_image, img_args), total=len(img_args), desc="Copying images"))

    # 复制标签（如果存在，多进程加速）
    if os.path.exists(src_label_dir):
        src_label_list = [f for f in os.listdir(src_label_dir) if f.endswith(".txt")]
        label_args = [(src_label_dir, dst_label_dir, label_file) for label_file in src_label_list]

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            list(tqdm(executor.map(copy_label, label_args), total=len(label_args), desc="Copying labels"))


def augment_single_image(args):
    img_file, src_img_dir, src_label_dir, dst_img_dir, dst_label_dir, rare_classes = args
    img_path = os.path.join(src_img_dir, img_file)
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(src_label_dir, label_file)

    image = cv2.imread(img_path)
    if image is None:
        return

    bboxes, class_labels = read_yolo_label(label_path)
    if len(bboxes) > 0:
        bboxes = np.array(bboxes, dtype=np.float32)
    else:
        bboxes = np.zeros((0, 4), dtype=np.float32)

    contains_rare = any(class_id in rare_classes for class_id in class_labels)
    num_augmentations = AUGMENT_COUNT + RARE_MORE_AUGMENT_COUNT if contains_rare else AUGMENT_COUNT
    is_small_image = image.shape[0] * image.shape[1] < 80 * 80

    for i in range(num_augmentations):
        augmentation = get_augmentation(extra=contains_rare, is_small=is_small_image)
        augmented = augmentation(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_image = augmented["image"]
        aug_bboxes = augmented["bboxes"]
        aug_class_labels = augmented["class_labels"]

        new_img_file = f"{os.path.splitext(img_file)[0]}_aug{i+1}{os.path.splitext(img_file)[1]}"
        new_label_file = f"{os.path.splitext(label_file)[0]}_aug{i+1}.txt"

        cv2.imwrite(os.path.join(dst_img_dir, new_img_file), aug_image)
        write_yolo_label(os.path.join(dst_label_dir, new_label_file), aug_bboxes, aug_class_labels)


def augment_batch_images(args_list):
    """
    批量增强图像（用于多进程加速）

    Args:
        args_list: 包含增强参数的列表
    """
    for args in args_list:
        augment_single_image(args)


def augment_train_set(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir):
    """
    对训练集进行数据增强（使用进程池并行加速）

    Args:
        src_img_dir: 源图像目录
        src_label_dir: 源标签目录
        dst_img_dir: 目标图像目录
        dst_label_dir: 目标标签目录
    """
    img_files = [f for f in os.listdir(src_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    class_counts = {i: 0 for i in range(10)}

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

    avg_count = sum(class_counts.values()) / len(class_counts)
    rare_classes = [class_id for class_id, count in class_counts.items() if count < avg_count]
    print(f"较少出现的数字 (将获得额外增强): {rare_classes}")

    args_list = [
        (img_file, src_img_dir, src_label_dir, dst_img_dir, dst_label_dir, rare_classes) for img_file in img_files
    ]

    if NUM_WORKERS <= 1:
        print("使用单线程增强图像...")
        for args in tqdm(args_list, desc="增强训练图像"):
            augment_single_image(args)
    else:
        print(f"使用 {NUM_WORKERS} 个工作进程增强图像...")
        
        # 使用多进程并行增强图像
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            list(tqdm(executor.map(augment_single_image, args_list), total=len(args_list), desc="增强训练图像"))

        # 如果需要分批处理，可以取消下面的注释
        # # 将任务划分为 NUM_WORKERS 个部分
        # chunk_size = len(args_list) // NUM_WORKERS + 1
        # args_chunks = [args_list[i : i + chunk_size] for i in range(0, len(args_list), chunk_size)]
        # for i, args_chunk in enumerate(args_chunks):
        #     print(f"处理批次 {i + 1}/{len(args_chunks)}，包含 {len(args_chunk)} 张图像")
        # # 使用多线程并行增强，每个线程处理一个批次
        # with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        #     list(tqdm(executor.map(augment_batch_images, args_chunks), total=len(args_chunks), desc="增强训练图像"))


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
    if not os.path.exists(os.path.join(USER_DATA_DIR, "data_augmented121.txt")):
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
