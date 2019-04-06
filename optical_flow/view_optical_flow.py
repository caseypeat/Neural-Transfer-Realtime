import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from tqdm import tqdm

dirpath = '../../Datasets/FlyingThings3D/FlyingThings3D_subset_image_clean/FlyingThings3D_subset/train/image_clean/right'
image_num = 100

image_array = np.zeros((image_num,540,960,3), dtype=np.uint8)

hsv = np.zeros((540,960,3), dtype=np.uint8)

for i, filename in tqdm(enumerate(os.listdir(dirpath)[:image_num]), total=image_num):

	filepath = os.path.join(dirpath, filename)

	image_array[i] = cv2.imread(filepath)

for i in range(image_num):

	image1 = cv2.cvtColor(image_array[i],cv2.COLOR_BGR2GRAY)
	image2 = cv2.cvtColor(image_array[i+1],cv2.COLOR_BGR2GRAY)

	flow = cv2.calcOpticalFlowFarneback(image1,image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

	plt.imshow(bgr)
	plt.show()