# Low-Pass Filtering + Inverse Filtering 실습 (주파수 영역에서의 흐림 및 복원)

from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import numpy.fft as fp
import matplotlib.pylab as pylab
import os

# 이미지 파일 경로 설정
image_path = '../image/lena.jpg'

if not os.path.exists(image_path):
    print(f"경고: {image_path} 파일을 찾을 수 없습니다. 코드를 종료합니다.")
    exit()
else:
    # 이미지 불러오기 (회색조 변환 후 0~255 스케일로)
    im = rgb2gray(imread(image_path)) * 255

# 가우시안 커널 파라미터
sigma = 3
size_x, size_y = im.shape

# 1D 가우시안 함수 수식 기반 커널 생성
center_x = size_x // 2
gauss_1d_x = np.exp(-np.power(np.arange(size_x) - center_x, 2.) / (2 * np.power(sigma, 2.)))
gauss_1d_x /= np.sum(gauss_1d_x)

center_y = size_y // 2
gauss_1d_y = np.exp(-np.power(np.arange(size_y) - center_y, 2.) / (2 * np.power(sigma, 2.)))
gauss_1d_y /= np.sum(gauss_1d_y)

# 2D 가우시안 커널 생성
gauss_kernel = np.outer(gauss_1d_x, gauss_1d_y)

# 원본 이미지 푸리에 변환
freq = fp.fft2(im)

# 커널 푸리에 변환 (ifftshift로 중심 정렬 후 FFT)
freq_kernel = fp.fft2(fp.ifftshift(gauss_kernel), s=im.shape)

# 주파수 영역에서 필터 적용
convolved = freq * freq_kernel
im_blur = fp.ifft2(convolved).real
im_blur = 255 * im_blur / np.max(im_blur)  # 정규화

# Inverse Filtering
epsilon = 1e-6  # 0으로 나누기 방지
freq_blur = fp.fft2(im_blur)
freq_kernel_inv = 1 / (epsilon + freq_kernel)
convolved_restored = freq_blur * freq_kernel_inv
im_restored = fp.ifft2(convolved_restored).real
im_restored = 255 * im_restored / np.max(im_restored)

# 시각화
pylab.figure(figsize=(10, 10))
pylab.gray()

pylab.subplot(2, 2, 1)
pylab.imshow(im)
pylab.title('Original Image')
pylab.axis('off')

pylab.subplot(2, 2, 2)
pylab.imshow(im_blur)
pylab.title('Blurred Image')
pylab.axis('off')

pylab.subplot(2, 2, 3)
pylab.imshow(im_restored)
pylab.title('Restored Image (Inverse Filter)')
pylab.axis('off')

pylab.subplot(2, 2, 4)
pylab.imshow(im_restored - im)
pylab.title('Difference (Restored - Original)')
pylab.axis('off')

pylab.tight_layout()
pylab.show()
