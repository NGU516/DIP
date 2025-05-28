import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from scipy.ndimage import uniform_filter, gaussian_filter, median_filter, generic_filter
import os

def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"경고: {image_path} 파일을 찾을 수 없습니다. 코드를 종료합니다.")
        return None
    # image to gray scale
    im = Image.open(image_path).convert('L')
    return np.array(im, dtype=np.float32)

# salt(255) and pepper(0), Random Noise
def add_salt_pepper_noise_to_array(image_array, proportion):
    noisy_image = np.copy(image_array)
    total_pixels = noisy_image.size
    num_noise_pixels = int(total_pixels * proportion)
    num_salt = int(num_noise_pixels * 0.5)
    coords_salt = [np.random.randint(0, dim, num_salt) for dim in noisy_image.shape]
    noisy_image[tuple(coords_salt)] = 255 if noisy_image.dtype == np.uint8 else 1.0
    num_pepper = num_noise_pixels - num_salt
    coords_pepper = [np.random.randint(0, dim, num_pepper) for dim in noisy_image.shape]
    noisy_image[tuple(coords_pepper)] = 0 if noisy_image.dtype == np.uint8 else 0.0
    return noisy_image

# frequency analysis, 2D Fourier Transform
def compute_frequency_spectrum(img):
    f = fft2(img)
    fshift = fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

# frequency analysis, radius energy distribution
def analyze_frequency_distribution(img):
    h, w = img.shape
    cx, cy = w // 2, h // 2     # mandrill image cx, cy = 112, 112
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)   # distance from center
    freq = fft2(img)                                # 2D Fourier Transform
    freq_shifted = fftshift(freq)                  # shift zero frequency component to center
    spectrum_magnitude = np.abs(freq_shifted)      # magnitude of spectrum
    r_max = int(distance.max())                     # max radius
    radius_energy = np.zeros(r_max + 1)            # initialize radius energy array
    # calculate radius energy
    for r in range(r_max + 1):
        mask = (distance >= r) & (distance < r + 1)   
        radius_energy[r] = spectrum_magnitude[mask].sum()
    normalized_energy = (radius_energy / np.sum(radius_energy)) * 100   # normalize energy
    cumulative_energy = np.cumsum(radius_energy)                        # cumulative energy
    normalized_cumulative = cumulative_energy / cumulative_energy[-1] * 100 # normalize cumulative energy
    return spectrum_magnitude, normalized_energy, normalized_cumulative

# frequency analysis, plot
def plot_frequency_analysis(img, normalized_energy, normalized_cumulative, spectrum_magnitude):
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(20 * np.log(np.abs(spectrum_magnitude) + 1), cmap='gray')
    plt.title('Frequency Spectrum')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.plot(normalized_energy)
    plt.title('Radius Energy Distribution (%)')
    plt.xlabel('Radius from Center')
    plt.ylabel('Energy (%)')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(normalized_cumulative)
    plt.title('Cumulative Energy Distribution (%)')
    plt.xlabel('Radius from Center')
    plt.ylabel('Cumulative Energy (%)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_pairwise_results(original, filtered, orig_title, filt_title):
    # 원본과 필터 이미지를 한 페이지에 비교
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    # 원본
    orig_spectrum, orig_norm_energy, _ = analyze_frequency_distribution(original)
    axs[0, 0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axs[0, 0].set_title(f'{orig_title} Image')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(20 * np.log(orig_spectrum + 1), cmap='gray')
    axs[0, 1].set_title('2D Fourier Spectrum')
    axs[0, 1].axis('off')
    axs[0, 2].plot(orig_norm_energy)
    axs[0, 2].set_title('Radius Energy Distribution (%)')
    axs[0, 2].set_xlabel('Radius from Center')
    axs[0, 2].set_ylabel('Energy (%)')
    axs[0, 2].grid(True)
    # 필터
    filt_spectrum, filt_norm_energy, _ = analyze_frequency_distribution(filtered)
    axs[1, 0].imshow(filtered, cmap='gray', vmin=0, vmax=255)
    axs[1, 0].set_title(f'{filt_title} Image')
    axs[1, 0].axis('off')
    axs[1, 1].imshow(20 * np.log(filt_spectrum + 1), cmap='gray')
    axs[1, 1].set_title('2D Fourier Spectrum')
    axs[1, 1].axis('off')
    axs[1, 2].plot(filt_norm_energy)
    axs[1, 2].set_title('Radius Energy Distribution (%)')
    axs[1, 2].set_xlabel('Radius from Center')
    axs[1, 2].set_ylabel('Energy (%)')
    axs[1, 2].grid(True)
    plt.tight_layout()
    plt.show()

def plot_all_results(images, titles):
    # images: [Original, S&P, mean, gauss, median, custom, highpass, ...]
    # titles: ["Original", ...]
    original = images[0]
    orig_title = titles[0]
    # Original vs S&P Noise
    plot_pairwise_results(original, images[1], orig_title, titles[1])
    # Original vs filtered
    for img, title in zip(images[2:], titles[2:]):
        plot_pairwise_results(original, img, orig_title, title)

# summarize interval energy, 10%, 10~20%, 20~30%, 30~50%, 50~100%
def summarize_interval_energy(cumulative, radii=[10, 20, 30, 50, 100]):
    prev = 0
    summary = {}
    for r in radii:
        if r < len(cumulative):
            summary[f"{prev}~{r}"] = cumulative[r] - cumulative[prev]
        else:
            summary[f"{prev}~{r}"] = cumulative[-1] - cumulative[prev]
        prev = r
    return summary

# Salt(255) and Pepper(0) exclude, return max
def custom_max_filter(image, size=3):
    def filter_func(values):
        center = values[len(values)//2]
        filtered = [v for v in values if v != 0 and v != 255]
        # if center is 0 or 255, return max(exclude 0 and 255)
        if center == 0 or center == 255:
            # if all values are 0 or 255, return 0
            if len(filtered) == 0:
                return 0  
            # if not all values are 0 or 255, return max(exclude 0 and 255)
            # center pixel is changed to max(exclude 0 and 255)
            return np.max(filtered)
        else:
            # if center is not 0 or 255, center pixel is not changed
            return center
    return generic_filter(image, filter_func, size=(size, size))

def main():
    image_path = r"image/mandrill.jpg"
    img = load_image(image_path)
    if img is None:
        return
    img_uint8 = img.astype(np.uint8)
    img_sp = add_salt_pepper_noise_to_array(img_uint8, proportion=0.05)
    img_mean = uniform_filter(img_sp.astype(float), size=3)
    img_gauss = gaussian_filter(img_sp.astype(float), sigma=1)
    img_median = median_filter(img_sp, size=3)
    img_custom = custom_max_filter(img_sp, size=3)
    images = [img, img_sp, img_mean, img_gauss, img_median, img_custom]
    titles = ['Original', 'Salt & Pepper Noise', 'Mean Filtered', 'Gaussian Filtered', 'Median Filtered', 'Custom Max Filter', 'Highpass Max Filter']
    plot_all_results(images, titles)

    # 에너지 분포 요약 및 표 출력
    radii = [10, 20, 30, 50, 100]
    summary_dict = {}
    for im, title in zip(images, titles):
        _, _, norm_cum = analyze_frequency_distribution(im)
        summary = summarize_interval_energy(norm_cum, radii)
        summary_dict[title] = summary

    # 표 헤더 출력
    header = ["{:>20}".format(" ")] + ["{:>10}".format(f"{r1}" if i == 0 else f"{r0}~{r1}") for i, (r0, r1) in enumerate(zip([0]+radii[:-1], radii))]
    print("\n===== 구간별 에너지 분포 요약표 (단위: %) =====")
    print("".join(header))
    print("-" * (22 + 12 * len(radii)))
    # 각 행 출력
    for title, summary in summary_dict.items():
        row = ["{:>20}".format(title)]
        for k in summary:
            row.append("{:10.2f}".format(summary[k]))
        print("".join(row))

if __name__ == "__main__":
    main()
