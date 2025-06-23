import pyfftw
from PIL import Image
import numpy as np

def momentum_to_real_space(input_image_path, output_image_path):
    # 读取动量空间图像
    img = Image.open(input_image_path).convert('L')
    data = np.array(img, dtype=np.float64)  # 使用 float64 提高精度
    # 反傅里叶变换到实空间
    fft_obj = pyfftw.builders.ifft2(data)
    real_space = np.abs(fft_obj())
    epsilon = 1e-10  # 防止 ptp 为零
    # 使用 fftshift 将低频分量移到中心
    real_space = np.fft.fftshift(real_space)
    # 归一化到0-65535 (16 位)
    real_space = (real_space - real_space.min()) / (np.ptp(real_space) + epsilon)
    real_space_img = Image.fromarray(np.uint16(real_space * 65535))  # 保存为 16 位图像
    real_space_img.save(output_image_path)

def real_to_momentum_space(input_image_path, output_image_path):
    # 读取实空间图像
    img = Image.open(input_image_path).convert('L')
    data = np.array(img, dtype=np.float64)  # 使用 float64 提高精度
    # 傅里叶变换到动量空间
    fft_obj = pyfftw.builders.fft2(data)
    momentum_space = np.abs(fft_obj())
    epsilon = 1e-10  # 防止 ptp 为零
    # 使用 fftshift 将低频分量移到中心
    momentum_space = np.fft.fftshift(momentum_space)
    # 归一化到0-65535 (16 位)
    momentum_space = (momentum_space - momentum_space.min()) / (np.ptp(momentum_space) + epsilon)
    momentum_space_img = Image.fromarray(np.uint16(momentum_space * 65535))  # 保存为 16 位图像
    momentum_space_img.save(output_image_path)

