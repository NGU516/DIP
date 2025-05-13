import numpy as np
from PIL import Image
import matplotlib.pylab as pylab
import os

# 이미지 파일 경로 설정
image_path = r"image\swans.jpg"

# 영상 표시 함수
def plot_image(image, title=''):
    pylab.title(title, size=14)
    pylab.imshow(image, cmap='gray')
    pylab.axis('off')

try:
    # 1. 이미지 불러오기 및 그레이스케일 변환
    im = Image.open(image_path).convert('L')
    img = np.array(im, dtype=np.float32)
    h, w = img.shape

    # 2. Floyd-Steinberg Dithering 적용
    for y in range(h):
        for x in range(w):
            old_pixel = img[y, x]
            new_pixel = 255 if old_pixel > 127 else 0  # 임계값 127로 이진화
            img[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            # 주변 픽셀에 오차 분산 (이미지 경계 체크)
            if x + 1 < w:
                img[y, x + 1] += quant_error * 7 / 16
            if y + 1 < h and x > 0:
                img[y + 1, x - 1] += quant_error * 3 / 16
            if y + 1 < h:
                img[y + 1, x] += quant_error * 5 / 16
            if y + 1 < h and x + 1 < w:
                img[y + 1, x + 1] += quant_error * 1 / 16

    # 3. 결과 이미지 변환 및 시각화
    img_result = np.clip(img, 0, 255).astype(np.uint8)
    im_result = Image.fromarray(img_result)

    pylab.figure(figsize=(12, 6))
    pylab.subplot(1, 2, 1)
    plot_image(im, 'Original Grayscale Image')
    pylab.subplot(1, 2, 2)
    plot_image(im_result, 'Floyd-Steinberg Dithered Image')
    pylab.tight_layout()
    pylab.show()

except FileNotFoundError as e:
    print(f"오류: 파일을 찾을 수 없습니다: {e.filename}")
except Exception as e:
    print(f"예상치 못한 오류가 발생했습니다: {e}")
