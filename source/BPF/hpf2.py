from PIL import Image
import matplotlib.pylab as pylab
import numpy as np
import numpy.fft as fp
from scipy import fftpack
import os

# 신호대잡음비 계산 함수
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)

# 이미지 경로 설정
image_path = '../image/cameraman.jpg'

# 이미지 불러오기 및 처리
if not os.path.exists(image_path):
    print(f"경고: {image_path} 파일을 찾을 수 없습니다. 코드를 종료합니다.")
    exit()
else:
    im = np.array(Image.open(image_path).convert('L'))  # 흑백으로 변환

# 푸리에 변환
freq = fp.fft2(im)
w, h = freq.shape
half_w, half_h = w // 2, h // 2  # DC 성분 위치

# 결과 저장
snrs_hp = []
lbs = list(range(1, 25))

# 시각화
pylab.figure(figsize=(20, 20))

for l in lbs:
    freq1 = np.copy(freq)
    freq2 = fftpack.fftshift(freq1)

    # 중심 l×l 영역 0으로 만들기 (저주파 제거)
    freq2[half_w - l:half_w + l + 1, half_h - l:half_h + l + 1] = 0

    # 역 푸리에 변환
    im1 = np.clip(fp.ifft2(fftpack.ifftshift(freq2)).real, 0, 255)

    # SNR 계산
    snrs_hp.append(signaltonoise(im1, axis=None))

    # 출력
    pylab.subplot(6, 4, l)
    pylab.imshow(im1, cmap='gray')
    pylab.title('F = ' + str(l + 1), size=16)
    pylab.axis('off')

# 여백 조정
pylab.subplots_adjust(wspace=0.1, hspace=0.2)
pylab.show()
