import numpy as np
from skimage import exposure, img_as_float
from skimage.color import rgb2gray
from skimage.io import imread
import matplotlib.pyplot as plt
import os

image_beans_path = './image/beans_g.png'
image_lena_path = './image/lena_g.png'


def plot_image(image, title, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.imshow(image, cmap='gray')
    ax.set_axis_off()
    ax.set_title(title, size=20)


def cdf(im):
    c, b = exposure.cumulative_distribution(im)
    full_bins = np.arange(256)
    full_cdf = np.interp(full_bins, b, c)

    return full_cdf

def hist_matching(c, c_t, im):
    pixels = np.arange(256)
    new_pixels = np.interp(c, c_t, pixels)
    im_matched = new_pixels[im.ravel()].reshape(im.shape)
    return im_matched.astype(np.uint8)

if not os.path.exists(image_beans_path):
    print(f"경고: {image_beans_path} 파일을 찾을 수 없습니다. 코드를 종료합니다.")
elif not os.path.exists(image_lena_path):
    print(f"경고: {image_lena_path} 파일을 찾을 수 없습니다. 코드를 종료합니다.")
else:

    im_loaded_beans = imread(image_beans_path)
    if im_loaded_beans.ndim == 3 and im_loaded_beans.shape[2] == 3:
        im = (rgb2gray(im_loaded_beans) * 255).astype(np.uint8)
    else:
        im = im_loaded_beans.astype(np.uint8)

    im_loaded_lena = imread(image_lena_path)
    if im_loaded_lena.ndim == 3 and im_loaded_lena.shape[2] == 3:
        im_t = (rgb2gray(im_loaded_lena) * 255).astype(np.uint8)
    else:
        im_t = im_loaded_lena.astype(np.uint8)

    c, c_t = cdf(im), cdf(im_t)

    im1 = hist_matching(c, c_t, im)
    c1 = cdf(im1)

    p = np.arange(256)
    plt.figure(figsize=(20, 12))
    plt.set_cmap('gray')

    ax1 = plt.subplot(2, 3, 1)
    plot_image(im, 'Input image', ax=ax1)

    ax2 = plt.subplot(2, 3, 2)
    plot_image(im_t, 'Template image', ax=ax2)

    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(p, c, 'r.-', label='input')
    ax3.plot(p, c_t, 'b.-', label='template')
    ax3.legend(prop={'size': 15})
    ax3.set_title('CDF (Input vs Template)', size=20)
    ax3.set_xlabel('Pixel Value')
    ax3.set_ylabel('CDF')

    ax4 = plt.subplot(2, 3, 4)
    plot_image(im1, 'Output image with Hist. Matching', ax=ax4)

    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(p, c, 'r.-', label='input')
    ax5.plot(p, c_t, 'b.-', label='template')
    ax5.plot(p, c1, 'g.-', label='output')
    ax5.legend(prop={'size': 15})
    ax5.set_title('CDF (Input, Template, Output)', size=20)
    ax5.set_xlabel('Pixel Value')
    ax5.set_ylabel('CDF')

    plt.tight_layout()
    plt.show()