import os
import cv2
import PIL.Image
import numpy as np
import imgviz

for d in ['train', 'val', 'test']:
	label_dir = 'REFUGE-FOVCrop-padding/{}/gt-ODOC/all'.format(d)
	output_label_dir = 'REFUGE-FOVCrop-padding/{}/ann'.format(d)
	os.makedirs(output_label_dir, exist_ok=True)
	for file in os.listdir(label_dir):
		print(file)
		label = cv2.imread(os.path.join(label_dir, file), flags=0)
		label[label==0] = 2
		label[label==255] = 0
		label[label==128] = 1
		label = PIL.Image.fromarray(label, mode='P')
		colormap = imgviz.label_colormap()
		label.putpalette(colormap)
		label.save(os.path.join(output_label_dir, file))