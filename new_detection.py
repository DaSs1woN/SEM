#coding:utf-8

import cv2
import numpy as np
import ImageTest
from matplotlib import pyplot as plt

imgpath='test.jpg'


def img_detection(img):
	height,width=img.shape[0:2]
	if height<=8:
		return 99999999
	result=ImageTest.atom_detection(img)
	for i,p in enumerate(result):
		if p["classes"] == 1:
			n=1
			while 1:
				new_img=img[n:height-n,n:width-n]
				result=ImageTest.atom_detection(new_img)
				for i,p in enumerate(result):
					if p["classes"]!=1:
						return (height+2-n*2)
				n+=1
		else:
			new_img1=img[0:int(height*0.5),0:int(width*0.5)]
#new_img2=img[int(height*0.5):height,int(width*0.5):width]
#new_img3=img[0:int(height*0.5),int(width*0.5):width]
			new_img4=img[int(height*0.5):height,0:int(width*0.5)]
			new_height1=img_detection(new_img1)
#new_height2=img_detection(new_img2)
#new_height3=img_detection(new_img3)
			new_height4=img_detection(new_img4)
			return min(new_height1,new_height4)


img=cv2.imread(imgpath,0)
height,width=img.shape[0:2]
print('height:',height)
print('width:',width)
#blur_img=cv2.blur(img,(5,5))
'''
circles=cv2.HoughCircles(blur_img,cv2.HOUGH_GRADIENT,1,20,param1=10,param2=10,minRadius=8,maxRadius=11)
print(circles)
print('nums:',len(circles[0]))
for circle in circles[0]:
	x=int(circle[0])
	y=int(circle[1])
	r=int(circle[2])
	img_cir=cv2.circle(img,(x,y),r,(255,0,0),1,8,0)
cv2.imshow("test",img_cir)
cv2.waitKey()
'''

detect_scale=img_detection(img)
print(detect_scale)

'''
fast=cv2.FastFeatureDetector_create(threshold=40,nonmaxSuppression=True,type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
keypoints=fast.detect(blur_img,None)
img=cv2.drawKeypoints(blur_img,keypoints,blur_img,color=(255,0,0))
cv2.imshow("test",img)
cv2.waitKey()
'''
