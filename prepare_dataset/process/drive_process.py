import cv2
import os
import numpy as np


for dataset in ['training','test']:
	img_dir = 'D:/dataset/Drive/{}/image_crop'.format(dataset)
	label_dir = 'D:/dataset/Drive/{}/ann_crop'.format(dataset)
	output_img_dir = 'D:/dataset/Drive/{}/image_pad'.format(dataset)
	output_label_dir = 'D:/dataset/Drive/{}/ann_pad'.format(dataset)

	os.makedirs(output_img_dir,exist_ok=True)
	os.makedirs(output_label_dir,exist_ok=True)

	for file in os.listdir(img_dir):
		print(dataset+'/'+file)
		img = cv2.imread(os.path.join(img_dir,file))
		label = cv2.imread(os.path.join(label_dir,file.replace('.tif','.png')),flags=0)
		h,w = img.shape[:2]
		if h != w:
			top,bottom,left,right = 0, 0, 0, 0
			if h > w:
				left = (h-w)//2
				right = (h-w)//2
				if (h-w)%2 != 0:
					right += 1
			else:
				top = (w-h)//2
				bottom = (w-h)//2
				if (w-h)%2 != 0:
					bottom += 1
			img = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=0)
			label = cv2.copyMakeBorder(label,top,bottom,left,right,cv2.BORDER_CONSTANT,value=0)
		#resized_img = cv2.resize(img,(1024,1024),interpolation=cv2.INTER_LINEAR)
		#resized_label = cv2.resize(label,(1024,1024),interpolation=cv2.INTER_NEAREST)
		cv2.imwrite(os.path.join(output_img_dir,file),img)
		cv2.imwrite(os.path.join(output_label_dir,file.replace('.jpg','.png')),label)

