# Wiener 필터 실습: 흐림 + 노이즈 영상 복원하기

from skimage import color, restoration
from skimage.io import imread
from scipy.signal import convolve2d as conv2
import numpy as np
import matplotlib.pylab as pylab
import os
import sys

# 이미지 경로
image_path = '../image/elephant_g.jpg'

# 파일 존재 확인
if not os.path.exists(image_path):
    print(f"경고: {image_path} 파일을 찾을 수 없습니다. 코드를 종료합니다.")
    sys.exit()
else:
    im = color.rgb2gray(imread(image_path))

# 흐림(평균 필터) + 노이즈 추가
n = 7
psf = np.ones((n, n)) / n**2
im1 = conv2(im, psf, 'same')
im1 += 0.1 * im1.std() * np.random.standard_normal(im1.shape)

# Wiener 필터 복원
im2, _ = restoration.unsupervised_wiener(im1, psf)

# 결과 시각화
fig, axs = pylab.subplots(1, 3, figsize=(16, 4))
axs[0].imshow(im, cmap='gray')
axs[0].axis('off')
axs[0].set_title('Original image', size=20)

axs[1].imshow(im1, cmap='gray')
axs[1].axis('off')
axs[1].set_title('Noisy blurred image', size=20)

axs[2].imshow(im2, cmap='gray')
axs[2].axis('off')
axs[2].set_title('Self tuned restoration', size=20)

fig.tight_layout()
pylab.gray()
pylab.show()
