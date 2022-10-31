import cv2
import os
import imgviz
import numpy as np
from PIL import Image

# RETA test dataset has no annotation
vessel_dir = 'train/vessel'
output_dir = 'train/ann'
os.makedirs(output_dir, exist_ok=True)

for img in os.listdir(vessel_dir):
    vessel = cv2.imread(os.path.join(vessel_dir, img), flags=0)
    ann = np.zeros(vessel.shape, dtype=np.uint8)
    ann[vessel>0] = 1
    label = Image.fromarray(ann, mode='P')
    colormap = imgviz.label_colormap()
    label.putpalette(colormap)
    filename = img.split('.')[0][:-7] + '.png'
    label.save(os.path.join(output_dir, filename))
