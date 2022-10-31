import cv2
import os
import numpy as np

path = 'D:/dataset/FOVCrop-padding/ARIA-FOVCrop'

for subset in ['AMD', 'Diabetic', 'Healthy']:
	img_dir = os.path.join(path, subset, 'images')
	BDP_dir = os.path.join(path, subset, 'gt-vessel-BDP')
	BSS_dir = os.path.join(path, subset, 'gt-vessel-BSS')
	output_img_dir = os.path.join(path, subset, 'images_pad')
	output_BDP_dir = os.path.join(path, subset, 'gt-vessel-BDP_pad')
	output_BSS_dir = os.path.join(path, subset, 'gt-vessel-BSS_pad')
	os.makedirs(output_img_dir, exist_ok=True)
	os.makedirs(output_BDP_dir, exist_ok=True)
	os.makedirs(output_BSS_dir, exist_ok=True)
	for root, dirs, files in os.walk(img_dir):
		for file in files:
			if not file.endswith('.tif'):
				continue
			print(os.path.join(root, file))
			img = cv2.imread(os.path.join(root, file))
			label_BDP = cv2.imread(os.path.join(BDP_dir, file.replace('.tif', '.png')), flags=0)
			label_BSS = cv2.imread(os.path.join(BSS_dir, file.replace('.tif', '.png')), flags=0)
			pad_img = cv2.copyMakeBorder(img,96,96,0,0,cv2.BORDER_CONSTANT,value=0)
			pad_label_BDP = cv2.copyMakeBorder(label_BDP,96,96,0,0,cv2.BORDER_CONSTANT,value=0)
			pad_label_BSS = cv2.copyMakeBorder(label_BSS,96,96,0,0,cv2.BORDER_CONSTANT,value=0)
			cv2.imwrite(os.path.join(output_img_dir,file), pad_img)
			cv2.imwrite(os.path.join(output_BDP_dir, file.replace('.tif', '.png')), pad_label_BDP)
			cv2.imwrite(os.path.join(output_BSS_dir, file.replace('.tif', '.png')), pad_label_BSS)
