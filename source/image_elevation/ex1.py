import numpy as np
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as compare_psnr # 이전 오류 수정 반영
from PIL import Image
import matplotlib.pylab as pylab
import os # 파일 경로 관련 모듈 import

# 이미지 파일 경로 설정
image_path = r"..\..\image\parrot.png"

# 영상 표시 함수
def plot_image(image, title=''):
    pylab.title(title, size=20)
    pylab.imshow(image)
    pylab.axis('off') # 그래프에 축 보이기 원한다면 이 라인 주석처리

# 히스토그램 그리기 함수
def plot_hist(r, g, b, title=''):
    r_ubyte, g_ubyte, b_ubyte = img_as_ubyte(r), img_as_ubyte(g), img_as_ubyte(b)
    r_flat, g_flat, b_flat = np.array(r_ubyte).ravel(), np.array(g_ubyte).ravel(), np.array(b_ubyte).ravel()
    pylab.hist(r_flat, bins=256, range=(0, 256), color='r', alpha=0.5)
    pylab.hist(g_flat, bins=256, range=(0, 256), color='g', alpha=0.5)
    pylab.hist(b_flat, bins=256, range=(0, 256), color='b', alpha=0.5)
    pylab.xlabel('pixel value', size=20)
    pylab.ylabel('frequency', size=20)
    pylab.title(title, size=20)

try:
    # 이미지 불러오기
    im = Image.open(image_path)
    im_r, im_g, im_b = im.split()

    pylab.style.use('ggplot')
    pylab.figure(figsize=(15, 5))
    pylab.subplot(121), plot_image(im, 'original image')
    pylab.subplot(122)
    plot_hist(im_r, im_g, im_b, 'histogram for RGB channels')
    pylab.show()

except FileNotFoundError:
    print(f"오류: {image_path} 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
except Exception as e:
    print(f"예상치 못한 오류가 발생했습니다: {e}")