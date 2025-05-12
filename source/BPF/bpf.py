# BPF 영상복원
# 70%, 80%, 90% 이상의 높은 주파수 성분을 통과
# 70~90% 대역 통과 필터
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import os

# 상대 경로로 이미지 파일 설정 (CV/source/BPF/bpf.py -> CV/image/moonlanding.png)
image_path = '../../image/moonlanding.png'

# 디버깅을 위한 경로 정보 출력
print("현재 작업 디렉토리:", os.getcwd())
print("현재 파일 위치:", os.path.abspath(__file__))
print("시도하는 이미지 경로:", os.path.abspath(image_path))

# SNR 계산 함수
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)

# 이미지 불러오기
if not os.path.exists(image_path):
    print(f"파일 {image_path} 을(를) 찾을 수 없습니다.")
    exit()

# 이미지 (이미 회색조라 처리X)
im = imread(image_path)

# 주파수 스펙트럼 시각화
freq = np.fft.fft2(im)
freq_shifted = np.fft.fftshift(freq)    # 주파수 정렬(DC 0,0 성분 중앙)
spectrum_magnitude = np.abs(freq_shifted)

plt.figure(figsize=(15, 10))
plt.imshow(np.log10(spectrum_magnitude), cmap='gray')
plt.title('Frequency Magnitude Spectrum (Center Zeroed)')
plt.colorbar()
plt.axis('off')
plt.show()

# 주파수 에너지 분포 분석 (반지름 기준)
h, w = im.shape
cx, cy = w // 2, h // 2
y, x = np.ogrid[:h, :w]
distance = np.sqrt((x - cx)**2 + (y - cy)**2)

r_max = int(distance.max())
radius_energy = np.zeros(r_max + 1)

# 각 주파수별 에너지의 총합
for r in range(r_max + 1):
    mask = (distance >= r) & (distance < r + 1)
    radius_energy[r] = spectrum_magnitude[mask].sum()

plt.figure(figsize=(15, 10))
normalized_energy = (radius_energy / np.sum(radius_energy)) * 100
plt.plot(normalized_energy)
plt.title('Radius Energy Distribution (%)')
plt.xlabel('Radius from Center')
plt.ylabel('Energy (%)')
plt.grid(True)
plt.show()

# 누적 에너지 계산
cumulative_energy = np.cumsum(radius_energy)
normalized_cumulative = cumulative_energy / cumulative_energy[-1] * 100 # % 계산

# 컷오프 반지름 (70%, 80%, 90%)
cutoff_percentages = [70, 80, 90]
cutoff_radius = [np.argmax(normalized_cumulative >= p) for p in cutoff_percentages]

# 누적 에너지 시각화
plt.figure(figsize=(15, 10))
plt.plot(normalized_cumulative)
plt.title('Cumulative Frequency Energy vs Radius')
plt.xlabel('Radius (Frequency Distance from Center)')
plt.ylabel('Cumulative Energy (%)')
plt.grid(True)

for p, r in zip(cutoff_percentages, cutoff_radius):
    plt.axvline(x=r, linestyle='--', label=f'{p}% → r ≥ {r}')

plt.legend()
plt.tight_layout()
plt.show()

# 기존 누적 에너지에서 원하는 % 구간 성분만 통과 (BPF)
# 70~80%, 80~90%, 70~90%
cutoff_percentages = [(0.7, 0.8), (0.8, 0.9), (0.7, 0.9)]  
bpf_images = []

for p_low, p_high in cutoff_percentages:
    r_low = np.argmax(normalized_cumulative >= p_low * 100)
    r_high = np.argmax(normalized_cumulative >= p_high * 100)

    # 대역 통과 마스크: r_low <= r < r_high
    mask = (distance >= r_low) & (distance < r_high)
    filtered_freq = freq_shifted * mask
    restored = np.fft.ifft2(np.fft.ifftshift(filtered_freq)).real
    bpf_images.append(((p_low, p_high), (r_low, r_high), restored))

# 결과 시각화
plt.figure(figsize=(16, 4))
plt.subplot(1, len(bpf_images) + 1, 1)
plt.imshow(im, cmap='gray')
plt.title('Original')
plt.axis('off')

for i, ((p1, p2), (r1, r2), img) in enumerate(bpf_images):
    plt.subplot(1, len(bpf_images) + 1, i + 2)
    plt.imshow(np.clip(img, 0, 255), cmap='gray')
    plt.title(f'{int(p1*100)}%~{int(p2*100)}%\n(r {r1}~{r2})')
    plt.axis('off')

plt.tight_layout()
plt.show()

print("--- 필터링 결과에 대한 SNR (Band-Pass Filter) ---")
for (p1, p2), (r1, r2), img in bpf_images:
    img_clipped = np.clip(img, 0, 255).astype(np.uint8)
    snr_val = signaltonoise(img_clipped, axis=None)
    print(f"주파수 대역 {int(p1*100)}% ~ {int(p2*100)}% (r: {r1} ~ {r2}) SNR = {snr_val:.4f}")




# Band-Stop Filter (BPF) 영상 복원 (진짜)
# 잡음 추정 반지름 범위
r_noise_min = 150
r_noise_max = 200

# 거리 기반 마스크 생성: 150 ≤ r < 200 영역만 제거 (Band-Stop)
mask = ~((distance >= r_noise_min) & (distance < r_noise_max))

# 필터링 및 복원
filtered_freq = freq_shifted * mask
restored = np.fft.ifft2(np.fft.ifftshift(filtered_freq)).real

# 시각화
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(im, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.clip(restored, 0, 255), cmap='gray')
plt.title(f'Filtered (r {r_noise_min}~{r_noise_max} filtering)')
plt.axis('off')

plt.tight_layout()
plt.show()

# SNR 계산
restored_clipped = np.clip(restored, 0, 255).astype(np.uint8)
snr_val = signaltonoise(restored_clipped, axis=None)
print(f"Filtered Image SNR (r = {r_noise_min}~{r_noise_max} filtering): {snr_val:.4f}")


