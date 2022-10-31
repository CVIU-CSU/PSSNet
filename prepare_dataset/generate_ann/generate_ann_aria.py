import os
import cv2
import PIL.Image
import numpy as np
import imgviz

for c in ['AMD', 'Diabetic', 'Healthy']:
	label_BDP_dir = 'ARIA-FOVCrop-padding/{}/gt-vessel-BDP'.format(c)   # change the path to yours
	label_BSS_dir = 'ARIA-FOVCrop-padding/{}/gt-vessel-BSS'.format(c)
	output_label_BDP_dir = 'ARIA-FOVCrop-padding/{}/ann-BDP'.format(c)
	output_label_BSS_dir = 'ARIA-FOVCrop-padding/{}/ann-BSS'.format(c)
	os.makedirs(output_label_BDP_dir, exist_ok=True)
	os.makedirs(output_label_BSS_dir, exist_ok=True)
	for file in os.listdir(label_BSS_dir):
		print(file)
		label_BDP = cv2.imread(os.path.join(label_BDP_dir, file), flags=0)
		label_BSS = cv2.imread(os.path.join(label_BSS_dir, file), flags=0)
		label_BDP[label_BDP>0] = 1
		label_BSS[label_BSS>0] = 1
		label_BDP = PIL.Image.fromarray(label_BDP, mode='P')
		label_BSS = PIL.Image.fromarray(label_BSS, mode='P')
		colormap = imgviz.label_colormap()
		label_BDP.putpalette(colormap)
		label_BSS.putpalette(colormap)
		label_BDP.save(os.path.join(output_label_BDP_dir, file))
		label_BSS.save(os.path.join(output_label_BSS_dir, file))