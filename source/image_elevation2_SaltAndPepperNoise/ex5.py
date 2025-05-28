import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageFilter
from skimage.color import rgb2gray

image_mandrill_path = '../image/mandrill.jpg'
output_dir = './image/'

def plot_image(image, title, ax=None):
    if ax is None:
        ax = plt.gca()
    if isinstance(image, Image.Image) and image.mode == 'L':
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)
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
    return noisy_image.astype(image_array.dtype)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(image_mandrill_path):
    print(f"경고: 원본 이미지 '{image_mandrill_path}'를 찾을 수 없습니다. 코드를 종료합니다.")
else:
    im_original_pil = Image.open(image_mandrill_path).convert('L') 

    current_subplot_index = 1
    plt.figure(figsize=(15, 10))

    for prop_noise in np.linspace(0.05, 0.3, 3):
        n_noise_pixels = int(im_original_pil.width * im_original_pil.height * prop_noise)
        im_array_for_noise = np.array(im_original_pil)
        im_noisy_array = add_salt_pepper_noise_to_array(im_array_for_noise, prop_noise)
        im_noisy_pil = Image.fromarray(im_noisy_array)
        output_filename = os.path.join(output_dir, f'mandrill_spnoise_{int(prop_noise*100)}%.jpg')
        im_noisy_pil.save(output_filename)
        title_noisy = f'Noisy ({int(100*prop_noise)}% S&P Noise)'
        plt.subplot(3, 2, current_subplot_index)
        plot_image(im_noisy_pil, title_noisy)
        current_subplot_index += 1
        im_mean_filtered_pil = im_noisy_pil.filter(ImageFilter.BLUR)         
        title_filtered = 'Mean Filtered'
        plt.subplot(3, 2, current_subplot_index)
        plot_image(im_mean_filtered_pil, title_filtered)
        current_subplot_index += 1
    plt.tight_layout()
    plt.show()