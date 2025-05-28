import matplotlib.pylab as pylab
import numpy as np
import os
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import exposure

image_path = './image/earthfromsky.jpg'

if not os.path.exists(image_path):
    print(f"경고: {image_path} 파일을 찾을 수 없습니다. 코드를 종료합니다.")
else:
    img = rgb2gray(imread(image_path))

    img_eq = exposure.equalize_hist(img)

    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    images = [img, img_eq, img_adapteq]
    titles = ['Original input (earth from sky)',
              'After histogram equalization',
              'After adaptive histogram equalization'] 

    pylab.gray()

    pylab.figure(figsize=(18, 12))

    for i in range(3):
        pylab.subplot(2, 3, i + 1)
        pylab.imshow(images[i], cmap='gray') 
        pylab.axis('off')
        pylab.title(titles[i], size=15)

        pylab.subplot(2, 3, i + 4) 
        pylab.hist(images[i].ravel(), bins=256, range=(0.0, 1.0), color='g', density=True)
        pylab.title('Histogram for ' + titles[i].split('(')[0].strip(), size=12) 
        pylab.xlabel('Pixel value'), pylab.ylabel('Normalized frequency')

    pylab.tight_layout()
    pylab.show()