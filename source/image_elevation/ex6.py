import numpy as np
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from PIL import Image, ImageEnhance
import matplotlib.pylab as pylab
import os

image_path = r"image\swans.jpg"

def plot_image(image, title=''):
    pylab.title(title, size=14)
    pylab.imshow(image, cmap='gray') # Gray 이미지이므로 cmap='gray' 설정
    pylab.axis('off')

def plot_hist(image, title=''): # Gray 이미지 히스토그램 처리
    img_array = np.array(image).ravel()
    pylab.hist(img_array, bins=256, range=(0, 256), color='gray')
    pylab.xlabel('Pixel values', size=10)
    pylab.ylabel('Frequency', size=10)
    pylab.title(title, size=12)

try:
    im_original = Image.open(image_path).convert('L')

    pylab.figure(figsize=(8, 6))
    plot_hist(im_original, 'Histogram of Original Grayscale Image')
    pylab.show()

    pylab.figure(figsize=(16, 12))
    pylab.subplot(221), plot_image(im_original, 'Original Grayscale Image')

    for i, th in enumerate([100, 150, 200]):
        im_binary = im_original.point(lambda x: 255 if x > th else 0)
        pylab.subplot(2, 2, i + 2),
        plot_image(im_binary, f'Binary Image (Threshold={th})')
    pylab.tight_layout()
    pylab.show()

except FileNotFoundError as e:
    print(f"오류: 파일을 찾을 수 없습니다: {e.filename}")
except Exception as e:
    print(f"예상치 못한 오류가 발생했습니다: {e}")