"""
获取数据集脚本

主要流程：
1. 下载原始数据集文件（包括训练集、验证集和测试集）
2. 解压缩下载的zip文件
3. 将数据转换为YOLO格式，包括训练集、验证集和测试集
4. 创建YOLO数据集配置文件
"""

import os
import requests
import zipfile
from tqdm import tqdm
import tempfile
import shutil
import json
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(BASE_DIR, "tcdata")
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")

# 确保数据目录存在
os.makedirs(USER_DATA_DIR, exist_ok=True)

# 创建数据目录
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
YOLO_DATASET_DIR = os.path.join(DATA_DIR, "yolo_dataset")
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(YOLO_DATASET_DIR, exist_ok=True)

# 数据集下载链接
datasets = {
    "mchar_train.zip": "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_train.zip",
    "mchar_train.json": "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_train.json",
    "mchar_val.zip": "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_val.zip",
    "mchar_val.json": "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_val.json",
    "mchar_test_a.zip": "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_test_a.zip",
    "mchar_sample_submit_A.csv": "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_sample_submit_A.csv",
}


def download_file(url, filename):
    """
    下载文件，带进度条。先下载到临时文件，下载完成后再移动到目标位置。
    如果中断，则删除临时文件。
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    tmp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(filename)) as tmp_file, tqdm(
            desc=f"正在下载 {os.path.basename(filename)}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            tmp_file_path = tmp_file.name
            for data in response.iter_content(chunk_size=1024):
                size = tmp_file.write(data)
                bar.update(size)
        shutil.move(tmp_file_path, filename)
    except BaseException as e:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        raise e


def extract_zip(filename, extract_dir):
    """解压zip文件"""
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"已解压 {filename} 到 {extract_dir}")


def download_raw_data(raw_data_dir):
    # 下载所有文件
    for filename, url in datasets.items():
        file_path = os.path.join(raw_data_dir, filename)

        if not os.path.exists(file_path):
            download_file(url, file_path)
        else:
            print(f"文件 {filename} 已存在，跳过下载。")

    # 解压所有zip文件
    for filename in datasets.keys():
        if filename.endswith(".zip"):
            zip_path = os.path.join(raw_data_dir, filename)
            extract_dir = os.path.join(raw_data_dir, filename.replace(".zip", ""))

            if not os.path.exists(extract_dir):
                print(f"正在解压 {filename} ...")
                os.makedirs(extract_dir, exist_ok=True)
                extract_zip(zip_path, raw_data_dir)
                macosx_dir = os.path.join(raw_data_dir, "__MACOSX")
                if os.path.exists(macosx_dir) and os.path.isdir(macosx_dir):
                    shutil.rmtree(macosx_dir)
                    print(f"已删除 {macosx_dir}")
            else:
                print(f"目录 {extract_dir} 已存在，跳过解压。")
    print("下载和解压已完成！")


def convert_to_yolo_format(input_dir, output_dir):
    """
    将SVHN数据集转换为YOLO格式
    """
    # 创建YOLO数据集目录结构
    yolo_train_dir = os.path.join(output_dir, "train")
    yolo_val_dir = os.path.join(output_dir, "val")
    yolo_test_dir = os.path.join(output_dir, "test")

    os.makedirs(yolo_train_dir, exist_ok=True)
    os.makedirs(yolo_val_dir, exist_ok=True)
    os.makedirs(yolo_test_dir, exist_ok=True)

    # 处理训练集
    train_json_path = os.path.join(input_dir, "mchar_train.json")
    train_img_dir = os.path.join(input_dir, "mchar_train")

    with open(train_json_path, "r") as f:
        train_data = json.load(f)

    # 处理验证集
    val_json_path = os.path.join(input_dir, "mchar_val.json")
    val_img_dir = os.path.join(input_dir, "mchar_val")

    with open(val_json_path, "r") as f:
        val_data = json.load(f)

    # 处理测试集
    test_img_dir = os.path.join(input_dir, "mchar_test_a")

    # 转换函数
    def convert_data(data_dict, img_dir, output_dir):
        # 创建输出目录
        img_output_dir = os.path.join(output_dir, "images")
        label_output_dir = os.path.join(output_dir, "labels")
        os.makedirs(img_output_dir, exist_ok=True)
        os.makedirs(label_output_dir, exist_ok=True)

        # 确保数据字典的格式正确
        for img_name, attrs in tqdm(data_dict.items(), desc=f"处理 {os.path.basename(output_dir)}"):
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                continue

            # 使用PIL读取图像尺寸
            with Image.open(img_path) as img:
                img_width, img_height = img.size

            yolo_lines = []
            # 按照新的数据结构，attrs中各字段为列表，通过zip遍历
            for label, left, top, w, h in zip(
                attrs["label"], attrs["left"], attrs["top"], attrs["width"], attrs["height"]
            ):
                # 归一化中心坐标和宽高
                x_center = (left + w / 2) / img_width
                y_center = (top + h / 2) / img_height
                norm_w = w / img_width
                norm_h = h / img_height
                yolo_lines.append(f"{label} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

            # 保存YOLO标注到txt文件
            txt_filename = os.path.splitext(img_name)[0] + ".txt"
            txt_path = os.path.join(label_output_dir, txt_filename)
            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_lines))

            # 复制图像到YOLO数据集目录
            shutil.copy(img_path, img_output_dir)

    # 转换训练集
    convert_data(train_data, train_img_dir, yolo_train_dir)

    # 转换验证集
    convert_data(val_data, val_img_dir, yolo_val_dir)

    # 处理测试集 - 只需要复制图像
    if os.path.exists(test_img_dir):
        img_output_dir = os.path.join(yolo_test_dir, "images")
        os.makedirs(img_output_dir, exist_ok=True)
        for img_name in tqdm(os.listdir(test_img_dir), desc="处理 test"):
            img_path = os.path.join(test_img_dir, img_name)
            if os.path.isfile(img_path):
                shutil.copy(img_path, img_output_dir)

    print("YOLO格式转换完成！")

    # 创建YOLO数据集配置文件
    create_yolo_config(yolo_dataset_dir=output_dir)


def create_yolo_config(yolo_dataset_dir):
    """
    创建YOLO数据集配置文件
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
    tag_file_path = os.path.join(USER_DATA_DIR, "data_downloaded.txt")
    if os.path.exists(tag_file_path):
        print(f"任务已完成标记文件已存在: {tag_file_path}")
        return
    with open(tag_file_path, "w") as f:
        f.write("done")
    print(f"任务已完成标记文件已创建: {tag_file_path}")


if __name__ == "__main__":
    if not os.path.exists(os.path.join(USER_DATA_DIR, "data_downloaded.txt")):
        print("开始下载和转换数据...")
        
        # 下载原始数据
        download_raw_data(RAW_DATA_DIR)
        if os.path.exists(YOLO_DATASET_DIR):
            print(f"已删除旧的YOLO数据集目录: {YOLO_DATASET_DIR}")
            shutil.rmtree(YOLO_DATASET_DIR)
        os.makedirs(YOLO_DATASET_DIR, exist_ok=True)
        
        # 转换为YOLO格式
        convert_to_yolo_format(RAW_DATA_DIR, YOLO_DATASET_DIR)
        completed_tag()
    else:
        print("数据已下载并转换为YOLO格式，跳过。")
