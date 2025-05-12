# HPF
import numpy as np
from skimage.io import imread
from scipy import signal
import matplotlib.pylab as pylab
import os

# SNR 계산 함수
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)

# 이미지 로드
image_path = '../image/lena.jpg'
if not os.path.exists(image_path):
    print(f"경고: {image_path} 파일을 찾을 수 없습니다. 코드를 종료합니다.")
    exit()

im = np.mean(imread(image_path), axis=2)

# Low-Pass Filter
freq_lp = np.fft.fft2(im)
size_lp = 11    # 필터 범위
sigma_lp = 3    # 표준편차, 흐림 정도
center_lp = size_lp // 2
gauss_1d = np.exp(-np.power(np.arange(size_lp) - center_lp, 2.) / (2 * np.power(sigma_lp, 2.)))
gauss_1d /= np.sum(gauss_1d)
gauss_kernel_lp = np.outer(gauss_1d, gauss_1d)
freq_kernel_lp = np.fft.fft2(np.fft.fftshift(gauss_kernel_lp), s=im.shape)
convolved_freq_lp = freq_lp * freq_kernel_lp
im_lp = np.fft.ifft2(convolved_freq_lp).real
im_lp_clipped = np.clip(im_lp, 0, 255).astype(np.uint8)

# High-Pass Filter (중심 저주파 제거)
freq_hp = np.fft.fft2(im)
freq_shifted_hp = np.fft.fftshift(freq_hp)
w_hp, h_hp = freq_hp.shape  # 이미지 크기
half_w_hp, half_h_hp = w_hp // 2, h_hp // 2
freq_shifted_hp[half_w_hp-10:half_w_hp+11, half_h_hp-10:half_h_hp+11] = 0
im_hp = np.fft.ifft2(np.fft.ifftshift(freq_shifted_hp)).real
im_hp_clipped = np.clip(im_hp, 0, 255).astype(np.uint8)

# SNR 출력
print("Signal-to-Noise Ratio (High-Pass 추정):", signaltonoise(im_hp_clipped, axis=None))

# 결과 시각화
pylab.figure(figsize=(15, 10))

pylab.subplot(2, 2, 1)
pylab.imshow(im, cmap='gray')
pylab.title('Original Image')
pylab.axis('off')

pylab.subplot(2, 2, 2)
pylab.imshow(im_lp_clipped, cmap='gray')
pylab.title('Low-Pass Filtered')
pylab.axis('off')

pylab.subplot(2, 2, 3)
pylab.imshow(im_hp_clipped, cmap='gray')
pylab.title('High-Pass Filtered (Center Zeroing)')
pylab.axis('off')

pylab.subplot(2, 2, 4)
pylab.imshow(20 * np.log10(0.1 + np.abs(np.fft.fftshift(freq_hp))).astype(int), cmap='viridis')
pylab.title('HPF Spectrum (Center Zeroed)')
pylab.axis('off')

pylab.tight_layout()
pylab.show()
