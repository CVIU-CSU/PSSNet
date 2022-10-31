import os
import cv2
import PIL.Image
import numpy as np
import imgviz

label_dir_ah = 'STARE-FOVCrop-padding/gt-vessel-ah'
label_dir_vk = 'STARE-FOVCrop-padding/gt-vessel-vk'
output_label_dir_ah = 'STARE-FOVCrop-padding/ann-ah'
output_label_dir_vk = 'STARE-FOVCrop-padding/ann-vk'

os.makedirs(output_label_dir_ah, exist_ok=True)
os.makedirs(output_label_dir_vk, exist_ok=True)

for file in os.listdir(label_dir_ah):
	print(file)
	label_ah = cv2.imread(os.path.join(label_dir_ah, file), flags=0)
	label_vk = cv2.imread(os.path.join(label_dir_vk, file), flags=0)
	label_ah[label_ah>0] = 1
	label_vk[label_vk>0] = 1
	label_ah = PIL.Image.fromarray(label_ah, mode='P')
	label_vk = PIL.Image.fromarray(label_vk, mode='P')
	colormap = imgviz.label_colormap()
	label_ah.putpalette(colormap)
	label_vk.putpalette(colormap)
	label_ah.save(os.path.join(output_label_dir_ah, file))
	label_vk.save(os.path.join(output_label_dir_vk, file))