# LPF
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

# 이미지 파일 경로 설정
image_path = '../image/rhino.jpg'

# 파일 존재 여부 확인
if not os.path.exists(image_path):
    print(f"경고: {image_path} 파일을 찾을 수 없습니다. 코드를 종료합니다.")
    exit()
else:
    # 흑백 변환 후 배열로 불러오기
    im = np.array(Image.open(image_path).convert('L'))

    # 푸리에 변환
    freq = fp.fft2(im)
    freq1 = np.copy(freq)
    freq_shifted = fp.fftshift(freq1)

    # 저주파 제거를 위한 마스크 생성
    freq_shifted_low = np.copy(freq_shifted)
    w, h = freq.shape
    half_w, half_h = w // 2, h // 2
    freq_shifted_low[half_w - 10:half_w + 11, half_h - 10:half_h + 11] = 0

    # 고주파만 남긴 주파수 구성
    freq_shifted_high = freq_shifted - freq_shifted_low

    # 역 푸리에 변환
    im1 = fp.ifft2(fp.ifftshift(freq_shifted_high)).real

    # SNR 출력
    print("Signal-to-Noise Ratio:", signaltonoise(im1, axis=None))

    # 결과 시각화
    pylab.imshow(im1, cmap='gray')
    pylab.axis('off')
    pylab.title('High-Pass Filtered Image')
    pylab.show()
