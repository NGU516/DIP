import numpy as np
from skimage import img_as_ubyte, img_as_float
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from PIL import Image
import matplotlib.pylab as pylab
import os

image_path = r"image\parrot.png"

def plot_image(image, title=''):
    pylab.title(title, size=14)
    pylab.imshow(image)
    pylab.axis('off')

def plot_hist(r, g, b, title=''):
    r_ubyte, g_ubyte, b_ubyte = img_as_ubyte(r), img_as_ubyte(g), img_as_ubyte(b)
    r_flat, g_flat, b_flat = np.array(r_ubyte).ravel(), np.array(g_ubyte).ravel(), np.array(b_ubyte).ravel()
    pylab.hist(r_flat, bins=256, range=(0, 256), color='r', alpha=0.5)
    pylab.hist(g_flat, bins=256, range=(0, 256), color='g', alpha=0.5)
    pylab.hist(b_flat, bins=256, range=(0, 256), color='b', alpha=0.5)
    pylab.xlabel('pixel value', size=10)
    pylab.ylabel('frequency', size=10)
    pylab.title(title, size=12)

try:
    im_original = Image.open(image_path)
    im_r_orig, im_g_orig, im_b_orig = im_original.split()

    im_log = im_original.point(lambda i: 255 * np.log(1 + i / 255))
    im_r_log, im_g_log, im_b_log = im_log.split()

    im_float = img_as_float(np.array(im_original))
    gamma = 1.5
    im_gamma = (im_float ** gamma)

    pylab.style.use('ggplot')
    pylab.figure(figsize=(18, 12))

    pylab.subplot(321), plot_image(im_original, 'Original Image')
    pylab.subplot(322)
    plot_hist(im_r_orig, im_g_orig, im_b_orig, 'Histogram (Original)')

    pylab.subplot(323), plot_image(im_log, 'Log Transformed Image')
    pylab.subplot(324)
    plot_hist(im_r_log, im_g_log, im_b_log, 'Histogram (Log Transform)')

    pylab.subplot(325), plot_image(im_gamma, f'Power-Law (Gamma={gamma}) Image')
    pylab.subplot(326)
    plot_hist(im_gamma[..., 0], im_gamma[..., 1], im_gamma[..., 2], f'Histogram (Gamma={gamma})')

    pylab.tight_layout()
    pylab.show()

except FileNotFoundError as e:
    print(f"오류: 파일을 찾을 수 없습니다: {e.filename}")
except Exception as e:
    print(f"예상치 못한 오류가 발생했습니다: {e}")