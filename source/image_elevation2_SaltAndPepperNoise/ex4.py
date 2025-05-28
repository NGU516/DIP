import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage.util import random_noise
from scipy.ndimage import uniform_filter
from skimage.color import rgb2gray 

image_path = './image/mandrill.jpg'

def plot_image(image, title, ax=None, cmap='gray'):
    if ax is None:
        ax = plt.gca()
    ax.imshow(image, cmap=cmap)
    ax.set_axis_off()
    ax.set_title(title, size=15)

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

if not os.path.exists(image_path):
    print(f"경고: 원본 이미지 '{image_path}'를 찾을 수 없습니다. 코드를 종료합니다.")
else:
    im_original_loaded = Image.open(image_path)
    if im_original_loaded.mode != 'L':
        im_original_pil = im_original_loaded.convert('L')
    else:
        im_original_pil = im_original_loaded

    im_float = np.array(im_original_pil).astype(float) / 255.0
    im_uint8 = np.array(im_original_pil).astype(np.uint8)
    noise_proportion = 0.05
    gaussian_var = 0.01    
    im_gaussian_noisy = random_noise(im_float, mode='gaussian', var=gaussian_var, clip=True)
    im_gaussian_mean_filtered = uniform_filter(im_gaussian_noisy, size=3)
    im_sp_noisy = add_salt_pepper_noise_to_array(im_uint8, proportion=noise_proportion)  
    im_sp_mean_filtered_float = uniform_filter(im_sp_noisy.astype(float) / 255.0, size=3)
    im_sp_mean_filtered = (im_sp_mean_filtered_float * 255).astype(np.uint8)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10)) # 2행 2열

    plot_image(im_gaussian_noisy, 'Gaussian Noise Image', ax=axs[0, 0])
    plot_image(im_gaussian_mean_filtered, 'Gaussian + Mean Filtered', ax=axs[0, 1])
    plot_image(im_sp_noisy, 'S&P Noise Image', ax=axs[1, 0])
    plot_image(im_sp_mean_filtered, 'S&P + Mean Filtered', ax=axs[1, 1])    
    plt.tight_layout()
    plt.show()