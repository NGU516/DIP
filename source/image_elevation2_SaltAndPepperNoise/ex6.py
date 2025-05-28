import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.io import imread
from skimage.color import rgb2gray
from scipy import ndimage

image_path = './image/mandrill_spnoise_0.1.jpg'

def plot_image(image, title, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.imshow(image, cmap='gray')
    ax.set_axis_off()
    ax.set_title(title, size=15)

if not os.path.exists(image_path):
    print(f"경고: 이미지 '{image_path}'를 찾을 수 없습니다. 코드를 종료합니다.")
else:
    im_loaded = imread(image_path)
    if im_loaded.ndim == 3:
        im = rgb2gray(im_loaded)
    else:
        im = im_loaded.astype(float)

    k, s = 7, 2
    im_box = ndimage.uniform_filter(im, size=(k, k))
    t = (((k - 1) / 2) - 0.5) / s
    im_gaussian = ndimage.gaussian_filter(im, sigma=(s, s), truncate=t)

    fig = plt.figure(figsize=(30, 10))
    plt.subplot(131)
    plot_image(im, 'Original Image')
    plt.subplot(132)
    plot_image(im_box, 'Filtered with Box Filter')
    plt.subplot(133)
    plot_image(im_gaussian, 'Filtered with Gaussian Filter')
    plt.tight_layout()
    plt.show()