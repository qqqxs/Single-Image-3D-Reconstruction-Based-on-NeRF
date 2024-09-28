import lpips
import torch
from PIL import Image
from torchvision import transforms


# 图片预处理
def image_preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((800, 800)),  # 调整图像大小
        transforms.ToTensor()  # 将图片转换为Tensor
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加批处理维度
    return image


# 加载预训练的LPIPS模型
lpips_model = lpips.LPIPS(net='alex')  # 使用AlexNet作为特征提取器

# 加载图片
image1 = image_preprocess('ref imgs/dragon/dragon.png')
image2 = image_preprocess('ref imgs/dragon/ref-coarse.png')
image3 = image_preprocess('ref imgs/dragon/ref-refine.png')
image4 = image_preprocess('ref imgs/dragon/df.png')
image5 = image_preprocess('ref imgs/dragon/pe.png')


# 计算LPIPS
distance_coarse = lpips_model(image1, image2)
distance_refine = lpips_model(image1, image3)
distance_df = lpips_model(image1, image4)
distance_pe = lpips_model(image1, image5)
print(f'Coarse LPIPS distance: {distance_coarse.item()}')
print(f'Refine LPIPS distance: {distance_refine.item()}')
print(f'DreamFusion LPIPS distance: {distance_df.item()}')
print(f'Point-E LPIPS distance: {distance_pe.item()}')

print("################################################################")
################################################################
import cv2
import numpy as np

# 读取原始图像和重建图像
original_image = cv2.imread('ref imgs/dragon/dragon.png')
reconstructed_image_coarse = cv2.imread('ref imgs/dragon/ref-coarse.png')
reconstructed_image_refine = cv2.imread('ref imgs/dragon/ref-refine.png')
reconstructed_image_df = cv2.imread('ref imgs/dragon/df.png')
reconstructed_image_pe = cv2.imread('ref imgs/dragon/pe.png')


original_image = cv2.resize(original_image, (800, 800))
reconstructed_image_coarse = cv2.resize(reconstructed_image_coarse, (800, 800))
reconstructed_image_refine = cv2.resize(reconstructed_image_refine, (800, 800))
reconstructed_image_df = cv2.resize(reconstructed_image_df, (800, 800))
reconstructed_image_pe = cv2.resize(reconstructed_image_pe, (800, 800))


# 计算均方误差（MSE）
mse1 = np.mean((original_image - reconstructed_image_coarse) ** 2)
mse2 = np.mean((original_image - reconstructed_image_refine) ** 2)
mse3 = np.mean((original_image - reconstructed_image_df) ** 2)
mse4 = np.mean((original_image - reconstructed_image_pe) ** 2)

# 计算PSNR
max_pixel_value = 255  # 对于8位图像，最大像素值为255
psnr1 = 10 * np.log10((max_pixel_value ** 2) / mse1)
psnr2 = 10 * np.log10((max_pixel_value ** 2) / mse2)
psnr3 = 10 * np.log10((max_pixel_value ** 2) / mse3)
psnr4 = 10 * np.log10((max_pixel_value ** 2) / mse4)

print(f"Coarse PSNR: {psnr1} dB")
print(f"Refine PSNR: {psnr2} dB")
print(f"DreamFusion PSNR: {psnr3} dB")
print(f"Point-E PSNR: {psnr4} dB")

print("################################################################")
################################################################
from skimage.metrics import structural_similarity as ssim
from skimage import io, transform

# 读取两幅图像
image1 = io.imread('ref imgs/dragon/dragon.png', as_gray=True)
image2 = io.imread('ref imgs/dragon/ref-coarse.png', as_gray=True)
image3 = io.imread('ref imgs/dragon/ref-refine.png', as_gray=True)
image4 = io.imread('ref imgs/dragon/df.png', as_gray=True)
image5 = io.imread('ref imgs/dragon/pe.png', as_gray=True)

# 将两幅图像调整到相同大小
image1 = transform.resize(image1, (800, 800), anti_aliasing=True)
image2 = transform.resize(image2, (800, 800), anti_aliasing=True)
image3 = transform.resize(image3, (800, 800), anti_aliasing=True)
image4 = transform.resize(image4, (800, 800), anti_aliasing=True)
image5 = transform.resize(image5, (800, 800), anti_aliasing=True)

# 数据类型转换为float32，确保数值精度
image1 = image1.astype(np.float32)
image2 = image2.astype(np.float32)
image3 = image3.astype(np.float32)
image4 = image4.astype(np.float32)
image5 = image5.astype(np.float32)

# 计算SSIM，为浮点图像指定数据范围
ssim_score_coarse = ssim(image1, image2, data_range=image1.max() - image1.min())
ssim_score_refine = ssim(image1, image3, data_range=image1.max() - image1.min())
ssim_score_df = ssim(image1, image4, data_range=image1.max() - image1.min())
ssim_score_pe = ssim(image1, image5, data_range=image1.max() - image1.min())

print(f"Coarse SSIM: {ssim_score_coarse}")
print(f"Refine SSIM: {ssim_score_refine}")
print(f"DreamFusion SSIM: {ssim_score_df}")
print(f"Point-E SSIM: {ssim_score_pe}")
