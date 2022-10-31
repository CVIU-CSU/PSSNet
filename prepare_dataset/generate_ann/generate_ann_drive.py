import os
import cv2
import PIL.Image
import numpy as np
import imgviz

for d in ['train', 'test']:
	label1_dir = 'DRIVE-FOVCrop-padding/{}/gt-vessel-1st'.format(d)
	output_label1_dir = 'DRIVE-FOVCrop-padding/{}/ann-1st'.format(d)
	os.makedirs(output_label1_dir, exist_ok=True)
	if d == 'test':
		label2_dir = 'DRIVE-FOVCrop-padding/{}/gt-vessel-2nd'.format(d)
		output_label2_dir = 'DRIVE-FOVCrop-padding/{}/ann-2nd'.format(d)
		os.makedirs(output_label2_dir, exist_ok=True)
	for file in os.listdir(label1_dir):
		print(file)
		label1 = cv2.imread(os.path.join(label1_dir, file), flags=0)
		label1[label1>0] = 1
		label1 = PIL.Image.fromarray(label1, mode='P')
		colormap = imgviz.label_colormap()
		label1.putpalette(colormap)
		label1.save(os.path.join(output_label1_dir, file))
		if d == 'test':
			label2 = cv2.imread(os.path.join(label2_dir, file), flags=0)
			label2[label2>0] = 1
			label2 = PIL.Image.fromarray(label2, mode='P')
			colormap = imgviz.label_colormap()
			label2.putpalette(colormap)
			label2.save(os.path.join(output_label2_dir, file))
