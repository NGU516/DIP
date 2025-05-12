import numpy as np
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from PIL import Image, ImageEnhance
import matplotlib.pylab as pylab
import os

image_path = r"image\cheetah.png"

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

def custom_contrast(c):
    return 0 if c < 70 else (255 if c > 150 else (255*c - 22950) / 48)

try:
    im_original = Image.open(image_path)
    im_r_orig, im_g_orig, im_b_orig = im_original.split()[:3]

    im_custom_contrast = im_original.point(custom_contrast)
    im_r_custom, im_g_custom, im_b_custom = im_custom_contrast.split()[:3]

    enhancer = ImageEnhance.Contrast(im_original)
    im_pil_contrast = enhancer.enhance(2)
    im_pil_np = np.array(im_pil_contrast).astype(np.uint8)
    if im_pil_np.ndim == 3:
        im_r_pil, im_g_pil, im_b_pil = im_pil_np[:,:,0], im_pil_np[:,:,1], im_pil_np[:,:,2]
    elif im_pil_np.ndim == 2:
        im_r_pil, im_g_pil, im_b_pil = im_pil_np, im_pil_np, im_pil_np
    elif im_pil_np.ndim == 4:
        im_r_pil, im_g_pil, im_b_pil = im_pil_np[:,:,0], im_pil_np[:,:,1], im_pil_np[:,:,2]
        im_pil_np = im_pil_np[:,:,:3]


    pylab.style.use('ggplot')

    # 원본 이미지 및 히스토그램 (Figure 1)
    pylab.figure(figsize=(15, 5))
    pylab.subplot(121), plot_image(im_original, 'Original Image')
    pylab.subplot(122), plot_hist(im_r_orig, im_g_orig, im_b_orig, 'Histogram (Original)')
    pylab.tight_layout()
    pylab.show()

    # 사용자 정의 콘트라스트 스트레칭 이미지 및 히스토그램 (Figure 2)
    pylab.figure(figsize=(15, 5))
    pylab.subplot(121), plot_image(im_custom_contrast, 'Custom Contrast Stretch')
    pylab.subplot(122), plot_hist(im_r_custom, im_g_custom, im_b_custom, 'Histogram (Custom Stretch)')
    pylab.tight_layout()
    pylab.show()

    # PIL ImageEnhance 모듈 사용 결과 이미지 및 히스토그램 (Figure 3)
    pylab.figure(figsize=(15, 5))
    pylab.subplot(121), plot_image(im_pil_np, 'PIL ImageEnhance (Contrast=2)')
    pylab.subplot(122), plot_hist(im_r_pil, im_g_pil, im_b_pil, 'Histogram (PIL Enhance)')
    pylab.yscale('log', base=10)
    pylab.tight_layout()
    pylab.show()

except FileNotFoundError as e:
    print(f"오류: 파일을 찾을 수 없습니다: {e.filename}")
except Exception as e:
    print(f"예상치 못한 오류가 발생했습니다: {e}")