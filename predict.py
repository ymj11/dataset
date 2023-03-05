
import os

import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries, slic

from unet1 import Unet

unet = Unet()

# while True:
imgs = os.listdir("./img/")
for jpg in imgs:

    img = Image.open("./img/" + jpg)
    r_image = unet.detect_image(img)
    r_image.save("./img_out/"+jpg)

print("end")
