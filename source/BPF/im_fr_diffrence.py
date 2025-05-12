# 영상 영역 vs 주파수 영역 컨볼루션 함수 실행시간 비교
from PIL import Image
from skimage.io import imread
from skimage.color import rgb2gray
from scipy import ndimage, signal
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
import matplotlib.pylab as pylab
import numpy as np
import time
import os

# lena 이미지 경로 (리눅스라면 절대경로 수정 필요)
image_path = '../image/lena.jpg'

if not os.path.exists(image_path):
    print(f"경고: {image_path} 파일을 찾을 수 없습니다")
    exit()

# 이미지 로드 후 흑백 변환
im = np.mean(imread(image_path), axis=2)

# 다양한 커널 크기 및 sigma 설정
kernel_sizes = [3, 7, 11, 21, 51]
sigmas = [1, 3, 5, 10, 20]

kernels = {}
for size, sigma in zip(kernel_sizes, sigmas):
    # 가우시안 커널 생성
    center = size // 2
    gauss_1d_centered = np.exp(-np.power(np.arange(size) - center, 2.) / (2 * np.power(sigma, 2.)))
    gauss_1d_centered /= np.sum(gauss_1d_centered)
    kernels[f'gaussian_{size}'] = np.outer(gauss_1d_centered, gauss_1d_centered)

    # 박스카 커널 생성
    boxcar_1d = uniform_filter1d(np.ones(size), size)
    boxcar_1d /= np.sum(boxcar_1d)
    kernels[f'boxcar_{size}'] = np.outer(boxcar_1d, boxcar_1d)

# 결과 저장용 딕셔너리
results = {'convolve': {}, 'fftconvolve': {}}
num_runs = 10

for kernel_name, kernel in kernels.items():
    print(f"Processing kernel: {kernel_name} (shape: {kernel.shape})")

    # convolve 실행 시간 측정
    start_time = time.time()
    for _ in range(num_runs):
        convolved_img_convolve = signal.convolve2d(im, kernel, mode='same', boundary='symm')
    end_time = time.time()
    results['convolve'][kernel_name] = (end_time - start_time) / num_runs
    print(f"  convolve average time: {results['convolve'][kernel_name]:.6f} seconds")

    # fftconvolve 실행 시간 측정
    start_time = time.time()
    for _ in range(num_runs):
        convolved_img_fftconvolve = signal.fftconvolve(im, kernel, mode='same')
    end_time = time.time()
    results['fftconvolve'][kernel_name] = (end_time - start_time) / num_runs
    print(f"  fftconvolve average time: {results['fftconvolve'][kernel_name]:.6f} seconds")
    print("-" * 40)

# 시각화
kernel_names = list(kernels.keys())
convolve_times = [results['convolve'][k] for k in kernel_names]
fftconvolve_times = [results['fftconvolve'][k] for k in kernel_names]
x = np.arange(len(kernel_names))
width = 0.35

fig, ax = pylab.subplots(figsize=(14, 6))
rects1 = ax.bar(x - width / 2, convolve_times, width, label='convolve')
rects2 = ax.bar(x + width / 2, fftconvolve_times, width, label='fftconvolve')

ax.set_ylabel('Average Time (seconds)')
ax.set_xlabel('Kernel (Size and Type)')
ax.set_title('Comparison of convolve and fftconvolve Execution Time')
ax.set_xticks(x)
ax.set_xticklabels([f"{k.split('_')[1]} ({k.split('_')[0]})" for k in kernel_names])
ax.legend()
fig.tight_layout()
pylab.show()
