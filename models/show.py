import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os

file1 = '../data/images/chanel/2007RST-CHANEL/img_001.jpg'
file2 = '../data/images/chanel/2007RST-CHANEL/img_002.jpg'
file3 = '../data/images/chanel/2007RST-CHANEL/img_004.jpg'
file4 = '../data/images/chanel/2007RST-CHANEL/img_005.jpg'
im1 = mpimg.imread(file1)
im2 = mpimg.imread(file2)
im3 = mpimg.imread(file3)
im4 = mpimg.imread(file4)

plt.subplot(1, 3, 1)
plt.imshow(im1)
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(im2)
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(im3)
plt.axis('off')
plt.subplots_adjust(wspace=0.02, hspace=0)
plt.show()