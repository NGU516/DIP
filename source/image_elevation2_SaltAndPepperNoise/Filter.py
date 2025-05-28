import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from scipy.ndimage import uniform_filter, gaussian_filter
import os

def load_image(image_path):
    """이미지를 로드하고 그레이스케일로 변환"""
    try:
        im = Image.open(image_path).convert('L')
        return np.array(im, dtype=np.float32)
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다: {image_path}")
        return None
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")
        return None

def add_salt_and_pepper_noise(image, amount=0.05):
    noisy = image.copy()
    num_salt = np.ceil(amount * image.size * 0.5).astype(int)
    num_pepper = np.ceil(amount * image.size * 0.5).astype(int)
    # Salt
    coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy[tuple(coords)] = 255
    # Pepper
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy[tuple(coords)] = 0
    return noisy

def compute_frequency_spectrum(img):
    """이미지의 주파수 스펙트럼 계산"""
    # FFT 적용
    f = fft2(img)
    # 주파수 스펙트럼을 중앙으로 이동
    fshift = fftshift(f)
    # 로그 스케일로 변환 (0으로 나누는 것을 방지하기 위해 1을 더함)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

def analyze_frequency_distribution(img):
    """이미지의 주파수 분포 분석"""
    h, w = img.shape
    cx, cy = w // 2, h // 2
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # FFT 및 스펙트럼 계산
    freq = fft2(img)
    freq_shifted = fftshift(freq)
    spectrum_magnitude = np.abs(freq_shifted)
    
    # 반지름별 에너지 분포 계산
    r_max = int(distance.max())
    radius_energy = np.zeros(r_max + 1)
    
    for r in range(r_max + 1):
        mask = (distance >= r) & (distance < r + 1)
        radius_energy[r] = spectrum_magnitude[mask].sum()
    
    # 정규화된 에너지 분포
    normalized_energy = (radius_energy / np.sum(radius_energy)) * 100
    
    # 누적 에너지 계산
    cumulative_energy = np.cumsum(radius_energy)
    normalized_cumulative = cumulative_energy / cumulative_energy[-1] * 100
    
    return spectrum_magnitude, normalized_energy, normalized_cumulative

def plot_frequency_analysis(img, normalized_energy, normalized_cumulative, spectrum_magnitude):
    """주파수 분석 결과 시각화"""
    plt.figure(figsize=(15, 10))
    
    # 원본 이미지
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # 주파수 스펙트럼
    plt.subplot(2, 2, 2)
    plt.imshow(20 * np.log(np.abs(spectrum_magnitude) + 1), cmap='gray')
    plt.title('Frequency Spectrum')
    plt.axis('off')
    
    # 반지름별 에너지 분포
    plt.subplot(2, 2, 3)
    plt.plot(normalized_energy)
    plt.title('Radius Energy Distribution (%)')
    plt.xlabel('Radius from Center')
    plt.ylabel('Energy (%)')
    plt.grid(True)
    
    # 누적 에너지 분포
    plt.subplot(2, 2, 4)
    plt.plot(normalized_cumulative)
    plt.title('Cumulative Energy Distribution (%)')
    plt.xlabel('Radius from Center')
    plt.ylabel('Cumulative Energy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_image_frequency(image_path):
    """이미지의 주파수 성분 분석 및 시각화"""
    # 이미지 로드
    img = load_image(image_path)
    if img is None:
        return
    
    # 주파수 분포 분석
    spectrum_magnitude, normalized_energy, normalized_cumulative = analyze_frequency_distribution(img)
    
    # 결과 시각화
    plot_frequency_analysis(img, normalized_energy, normalized_cumulative, spectrum_magnitude)
    
    # 주파수 성분의 통계적 분석
    print("\n주파수 성분 분석:")
    print(f"주파수 스펙트럼 평균: {np.mean(spectrum_magnitude):.2f}")
    print(f"주파수 스펙트럼 표준편차: {np.std(spectrum_magnitude):.2f}")
    print(f"최대 에너지 반지름: {np.argmax(normalized_energy)}")
    print(f"90% 에너지 누적 반지름: {np.argmax(normalized_cumulative >= 90)}")

def plot_all_results(images, titles):
    n = len(images)
    page_size = 2
    num_pages = (n + page_size - 1) // page_size
    for page in range(num_pages):
        plt.figure(figsize=(12, 8))
        for i in range(page_size):
            idx = page * page_size + i
            if idx >= n:
                break
            img, title = images[idx], titles[idx]
            spectrum, norm_energy, cum_energy = analyze_frequency_distribution(img)
            # 원본/필터/노이즈 이미지
            plt.subplot(page_size, 3, i * 3 + 1)
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
            plt.title(f'{title}\nImage')
            plt.axis('off')
            # 2D 푸리에 스펙트럼
            plt.subplot(page_size, 3, i * 3 + 2)
            plt.imshow(20 * np.log(spectrum + 1), cmap='gray')
            plt.title('2D Fourier Spectrum')
            plt.axis('off')
            # 반지름별 에너지 분포
            plt.subplot(page_size, 3, i * 3 + 3)
            plt.plot(norm_energy)
            plt.title('Radius Energy Distribution (%)')
            plt.xlabel('Radius from Center')
            plt.ylabel('Energy (%)')
            plt.grid(True)
        plt.tight_layout()
        plt.show()

def main():
    image_path = r"image/mandrill.jpg"
    img = load_image(image_path)
    if img is None:
        return
    # Salt and Pepper Noise
    img_sp = add_salt_and_pepper_noise(img, amount=0.05)
    # Mean Filter
    img_mean = uniform_filter(img_sp, size=3)
    # Gaussian Filter
    img_gauss = gaussian_filter(img_sp, sigma=1)
    images = [img, img_sp, img_mean, img_gauss]
    titles = ['Original', 'Salt & Pepper Noise', 'Mean Filtered', 'Gaussian Filtered']
    plot_all_results(images, titles)

if __name__ == "__main__":
    main()
