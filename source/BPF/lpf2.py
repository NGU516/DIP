# Low-Pass Filter (저역 통과 필터)

from PIL import Image
import matplotlib.pylab as pylab
import numpy as np
import numpy.fft as fp
import os

# 신호대잡음비 함수 정의
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)

# 이미지 파일 경로
image_path = '../image/cameraman.jpg'

# 파일 확인 및 불러오기
if not os.path.exists(image_path):
    print(f"경고: {image_path} 파일을 찾을 수 없습니다. 코드를 종료합니다.")
    exit()
else:
    im = np.array(Image.open(image_path).convert('L'))

# 푸리에 변환
freq = fp.fft2(im)
w, h = freq.shape
half_w, half_h = w // 2, h // 2  # 중심 좌표

# 결과 저장용
snrs_lp = []
ubs = list(range(1, 31))

# 시각화
pylab.figure(figsize=(20, 20))

for u in ubs:
    freq_shifted = fp.fftshift(np.copy(freq))

    # 저역 통과 마스크 생성
    mask = np.zeros_like(freq_shifted, dtype=bool)
    mask[half_w - u : half_w + u + 1, half_h - u : half_h + u + 1] = True
    freq_shifted_low = freq_shifted * mask

    # 역변환 및 실수부 추출
    im1 = fp.ifft2(fp.ifftshift(freq_shifted_low)).real
    snrs_lp.append(signaltonoise(im1, axis=None))

    # 이미지 출력
    pylab.subplot(6, 5, u)
    pylab.imshow(im1, cmap='gray')
    pylab.axis('off')
    pylab.title('F = ' + str(u), size=16)

# 레이아웃 정리
pylab.subplots_adjust(wspace=0.1, hspace=0.2)
pylab.show()
