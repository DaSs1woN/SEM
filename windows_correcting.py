#coding:utf-8

import cv2
import numpy as np
import ImageTest
import math
import matplotlib.pyplot as plt

imgpath='new_atom_images/BFO-NbSTO-3.tif'
img=cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
height,width=img.shape[0:2]

w_pace=input('水平移动步长')
h_pace=input('垂直移动步长')

in_w=input('初始横坐标')
in_h=input('初始纵坐标')

window=input('窗口尺寸')
half_window=int(window*0.5)
half_pace=8

def rotate(angle,x,y,r):
	x1=x
	y1=height-y
	rotatex=r*math.cos(angle)+x1
	rotatey=r*math.sin(angle)+y1
	new_x,new_y=rotatex,height-rotatey
	return int(new_x),int(new_y)

def shrinking(w,h):
	w=w-half_window
	h=h-half_window
	r=1
	while r<half_window:
		img_tmp=img[h-half_window+r:h+half_window-r,w-half_window+r:w+half_window-r]
		result=ImageTest.atom_detection(img_tmp)
		for i,p in enumerate(result):
			if p["classes"]!=1:
				cv2.rectangle(img,(w-half_window+r-1,h-half_window+r-1),(w+half_window-r+1,h+half_window-r+1),(255,0,0),1)
				return
			else:
				r+=1
	return

def correcting(w,h):
	if half_pace+window<w<width-half_pace and half_pace+window<h<height-half_pace:
		in_angle=0
		lim_angle=360
	elif w<=half_pace+window and h<=half_pace+window:
		in_angle=-90
		lim_angle=0
	elif w<=half_pace+window and half_pace+window<h<height-half_pace:
		in_angle=-90
		lim_angle=90
	elif w>=width-half_pace and half_pace+window<h<height-half_pace:
		in_angle=90
		lim_angle=270
	elif half_pace+window<w<width-half_pace and h<=half_pace+window:
		in_angle=180
		lim_angle=360
	elif half_pace+window<w<width-half_pace and h>=height-half_pace:
		in_angle=0
		lim_angle=180
	else:
		in_angle=90
		lim_angle=180
	w=w-half_window
	h=h-half_window
	for i in range(1,half_pace):
		angle=in_angle
		while angle<lim_angle:
			angle_=math.radians(angle)
			new_w,new_h=rotate(angle_,w,h,i)
			new_img=img[new_h-half_window:new_h+half_window,new_w-half_window:new_w+half_window]
			result=ImageTest.atom_detection(new_img)
			for j,p in enumerate(result):
				if p["classes"] == 1:
					return new_w+half_window,new_h+half_window
			angle+=15
	return w+half_window,h+half_window


h,w=in_h+window,in_w+window
while h<height:
	while w<width:
		tmp=img[h-window:h,w-window:w]
		result=ImageTest.atom_detection(tmp)
		for i,p in enumerate(result):
			if p["classes"] == 1:
				shrinking(w,h)
#cv2.rectangle(img,(w-window,h-window),(w,h),(255,0,0),1)
			
			else:
#if half_pace+window<w<width-half_pace and half_pace+window<h<height-half_pace:
				tmp_w,tmp_h=w,h
				w,h=correcting(w,h)
				if w==tmp_w and h==tmp_h:
					print("**********************************************************")
				else:
					shrinking(w,h)
#cv2.rectangle(img,(w-window,h-window),(w,h),(255,0,0),1)
			
		w+=w_pace
	h+=h_pace
	w=in_w+window
cv2.imwrite('detection_test_7.jpg',img)
cv2.imshow('test',img)
cv2.waitKey()

