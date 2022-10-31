import cv2
import os
import numpy as np


for dataset in ['train']:
	img_dir = 'D:/dataset/refuge/image/{}'.format(dataset)
	label_dir = 'D:/dataset/refuge/disc_cup/{}'.format(dataset)
	output_img_dir = 'D:/dataset/refuge/image/{}_pad'.format(dataset)
	output_label_dir = 'D:/dataset/refuge/disc_cup/{}_pad'.format(dataset)

	os.makedirs(output_img_dir,exist_ok=True)
	os.makedirs(output_label_dir,exist_ok=True)

	for file in os.listdir(img_dir):
		print(file)
		img = cv2.imread(os.path.join(img_dir,file))
		label = cv2.imread(os.path.join(label_dir,file.replace('.jpg','.png')),flags=0)
		if dataset == 'train':
			img = cv2.copyMakeBorder(img,34,34,0,0,cv2.BORDER_CONSTANT,value=0)   
			label = cv2.copyMakeBorder(label,34,34,0,0,cv2.BORDER_CONSTANT,value=0)
		#resized_img = cv2.resize(img,(1024,1024),interpolation=cv2.INTER_LINEAR)
		#resized_label = cv2.resize(label,(1024,1024),interpolation=cv2.INTER_NEAREST)
		cv2.imwrite(os.path.join(output_img_dir,file), img)
		cv2.imwrite(os.path.join(output_label_dir,file.replace('.jpg','.png')), label)

