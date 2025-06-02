# 这里的是实验性代码，目前未加入到主流程中

import os
import cv2
import matplotlib.pyplot as plt

import numpy as np
from skimage import exposure

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(BASE_DIR, "tcdata")
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")

# 原始YOLO数据集目录
ORIGINAL_DATASET_DIR = os.path.join(DATA_DIR, "yolo_dataset_simple/train/images")

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from basicsr.utils.download_util import load_file_from_url


def realesrgan_upscale(
    img_path,
    model_name="RealESRGAN_x4plus_anime_6B",
    outscale=4,
    tile=400,  # 显存不足时减小此值
    gpu_id=None,
):
    """Real-ESRGAN超分辨率增强

    Args:
        img_path (str): 输入图片路径
        model_name (str): 模型名称，可选:
            'RealESRGAN_x4plus' (默认),
            'RealESRNet_x4plus',
            'RealESRGAN_x4plus_anime_6B',
            'RealESRGAN_x2plus'
        outscale (int): 输出放大倍数
        tile (int): 分块处理大小(0为禁用)
        gpu_id (int): 指定GPU ID

    Returns:
        np.ndarray: 超分后的BGR图像
    """
    # 模型选择
    model_dict = {
        "RealESRGAN_x4plus": {
            "model": RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            ),
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        },
        "RealESRGAN_x4plus_anime_6B": {
            "model": RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=6,
                num_grow_ch=32,
                scale=4,
            ),
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        },
        "RealESRGAN_x2plus": {
            "model": RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            ),
            "url": "https://github.com/xinntao/RealESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        },
        
    }

    # 初始化模型
    model = model_dict[model_name]["model"]
    upsampler = RealESRGANer(
        scale=4 if "x4" in model_name else 2,
        model_path=load_file_from_url(model_dict[model_name]["url"], "weights"),
        model=model,
        tile=tile,
        gpu_id=gpu_id,
    )

    # 读取并处理图像
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")

    # 执行超分
    output, _ = upsampler.enhance(img, outscale=outscale)
    return output


def process_and_display_images(image_paths, rows=5, cols=3):
    """
    处理并展示图片：局部对比度增强(CLAHE)、背景抑制(导向滤波)、光照校正(替代方案)

    参数:
        image_paths (list): 图片路径列表
        rows (int): 每轮展示的行数（默认4行：原图+3种处理结果）
        cols (int): 每轮展示的列数（默认3张图片）
    """
    # 初始化处理算法
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def guided_filter(img):
        return cv2.ximgproc.guidedFilter(img, img, radius=10, eps=1e-3)

    def illumination_correction(img):
        """替代的光照校正方案（三选一）"""
        # # 方案1：对数变换+归一化
        # corrected = np.log1p(img.astype(np.float32))
        # corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
        # return corrected.astype(np.uint8)

        # # 方案2：同态滤波（需取消下面注释）
        # gray_float = img.astype(np.float32) / 255.0
        # rows, cols = img.shape
        # crow, ccol = rows//2, cols//2
        # dft = cv2.dft(gray_float, flags=cv2.DFT_COMPLEX_OUTPUT)
        # dft_shift = np.fft.fftshift(dft)
        # mask = np.zeros((rows, cols, 2), np.float32)
        # mask[crow-30:crow+30, ccol-30:ccol+30] = 1
        # fshift = dft_shift * mask
        # f_ishift = np.fft.ifftshift(fshift)
        # img_back = cv2.idft(f_ishift)
        # img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        # return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 方案3：自适应阈值法（需取消下面注释）
        return cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10
        )

    for i in range(0, len(image_paths), cols):
        batch_paths = image_paths[i : i + cols]
        if not batch_paths:
            break

        plt.figure(figsize=(15, 10))

        for j, img_path in enumerate(batch_paths):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: 无法读取图片 {img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 处理步骤
            clahe_img = clahe.apply(gray)
            guided_img = guided_filter(gray)
            corrected_img = illumination_correction(gray)
            super_res_img = realesrgan_upscale(img_path)  # 假设有super_res函数

            # 显示结果
            plots = [
                ("Original", gray),
                ("CLAHE Enhanced", clahe_img),
                ("Background Suppressed", guided_img),
                ("Illumination Corrected", corrected_img),
                ("SuperResolution", super_res_img),  # 假设有super_res_img变量
            ]

            for k, (title, result_img) in enumerate(plots):
                plt.subplot(rows, cols, j + 1 + k * cols)
                plt.imshow(result_img, cmap="gray")
                plt.title(title if j == 0 else "")  # 只在第一列显示标题
                plt.axis("off")

        plt.tight_layout()
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 替换为你的图片路径列表
    # example_paths = [
    #     "sample1.jpg",
    #     "sample2.jpg",
    #     "sample3.jpg",
    #     # 添加更多图片路径...
    # ]

    example_paths = [
        os.path.join(ORIGINAL_DATASET_DIR, f)
        for f in os.listdir(ORIGINAL_DATASET_DIR)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    process_and_display_images(example_paths)
