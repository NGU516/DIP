import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import exposure, img_as_float

image_path = './image/beans_g.png'



def plot_image_and_hist(image, axes, bins=256):
    image = img_as_float(image)
    axes_image, axes_hist = axes
    axes_cdf = axes_hist.twinx()
    axes_image.imshow(image, cmap='gray')
    axes_image.set_axis_off()
    axes_hist.hist(image.ravel(), bins=bins, histtype='step', color='black', range=(0.0, 1.0), density=True)
    axes_hist.set_xlim(0, 1)
    axes_hist.set_xlabel('Pixel intensity', size=15)
    axes_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    axes_hist.set_yticks([])
    image_cdf, bins = exposure.cumulative_distribution(image, bins)
    axes_cdf.plot(bins, image_cdf, 'r')
    axes_cdf.set_yticks([])
    return axes_image, axes_hist, axes_cdf

if not os.path.exists(image_path):
    print(f"경고: {image_path} 파일을 찾을 수 없습니다. 코드를 종료합니다.")
else:

    im_loaded = imread(image_path)
    if im_loaded.ndim == 3 and im_loaded.shape[2] == 3:
        im = rgb2gray(im_loaded)
    else:
        im = im_loaded

    im_float = img_as_float(im)


    p2, p98 = np.percentile(im_float, (2, 98))
    im_rescale = exposure.rescale_intensity(im_float, in_range=(p2, p98))

    im_eq = exposure.equalize_hist(im_float)

    im_adapteq = exposure.equalize_adapthist(im_float, clip_limit=0.03)

    plt.rcParams['font.size'] = 8

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 7))
    axes_image, axes_hist, axes_cdf = plot_image_and_hist(im_float, axes[:, 0])
    axes_image.set_title('Low contrast image', size=20)
    y_min, y_max = axes_hist.get_ylim()
    axes_hist.set_ylabel('Number of pixels', size=15)
    axes_hist.set_yticks(np.linspace(0, y_max, 5))

    axes_image, axes_hist, axes_cdf = plot_image_and_hist(im_rescale, axes[:, 1])
    axes_image.set_title('Contrast stretching', size=20)

    axes_image, axes_hist, axes_cdf = plot_image_and_hist(im_eq, axes[:, 2])
    axes_image.set_title('Histogram equalization', size=20)


    axes_image, axes_hist, axes_cdf = plot_image_and_hist(im_adapteq, axes[:, 3])
    axes_image.set_title('Adaptive equalization', size=20)
    axes_cdf.set_ylabel('Fraction of total intensity', size=15)
    axes_cdf.set_yticks(np.linspace(0, 1, 5))

    fig.tight_layout()
    plt.show()