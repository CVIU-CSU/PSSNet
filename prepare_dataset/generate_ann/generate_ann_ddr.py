import os
from PIL import Image
import cv2
import numpy as np
import imgviz


for d in ['train','val','test']:
	path = 'DDR-FOVCrop-padding/{}/images'.format(d)
	output_dir = 'DDR-FOVCrop-padding/{}/ann'.format(d)
	os.makedirs(output_dir, exist_ok=True)
	for file in os.listdir(path):
		print(os.path.join(path,file))
		img = cv2.imread(os.path.join(path,file), flags=0)
		ann = np.zeros(img.shape,dtype=np.int32)
		filename = file.split('.')[0]
		for i,c in enumerate(['EX','HE','SE','MA']):
			label_dir = 'DDR-FOVCrop-padding/{}/gt-lesions/{}'.format(d,c)
			label = cv2.imread(os.path.join(label_dir, filename+'.png'), flags=0)
			ann[label>0] = i + 1
		ann_pil = Image.fromarray(ann.astype(np.uint8), mode="P")
		colormap = imgviz.label_colormap()
		ann_pil.putpalette(colormap)
		ann_pil.save(os.path.join(output_dir, filename+'.png'))
