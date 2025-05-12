import numpy as np
from PIL import Image
import matplotlib.pylab as pylab
import os

image_path = r"..\..\image\swans.jpg"

def plot_image(image, title=''):
    pylab.title(title, size=14)
    pylab.imshow(image)
    pylab.gray()
    pylab.axis('off')

try:
    im = Image.open(image_path).convert('L')
    im_noisy = np.array(im) + np.random.randint(-128, 128, (im.height, im.width))
    im_noisy = Image.fromarray(np.clip(im_noisy, 0, 255).astype(np.uint8))

    pylab.figure(figsize=(16,12))
    pylab.subplot(221), plot_image(im_noisy, 'original image (with noise)')

    for i, th in enumerate([100, 150, 200]):
        im1 = im_noisy.point(lambda x: 255 if x > th else 0)
        pylab.subplot(2,2,i+2)
        plot_image(im1, 'binary image with threshold=' + str(th))
    pylab.tight_layout()
    pylab.show()

except FileNotFoundError as e:
    print(f"오류: 파일을 찾을 수 없습니다: {e.filename}")
except Exception as e:
    print(f"예상치 못한 오류가 발생했습니다: {e}")