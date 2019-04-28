# coding=UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import cv2
import random

imgpath="testimg/"

def cnn_model_fn(features,labels,mode):
    input_layer=tf.reshape(features["x"],[-1,28,28,1])

    conv1=tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu)

    pool1=tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2,2],
            strides=2)

    conv2=tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu)
    pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)

    pool2_flat=tf.reshape(pool2,[-1,7*7*64])
    dense=tf.layers.dense(inputs=pool2_flat,units=1024,activation=tf.nn.relu)
    dropout=tf.layers.dropout(inputs=dense,rate=0.4,training=mode==tf.estimator.ModeKeys.TRAIN)
    logits=tf.layers.dense(inputs=dropout,units=6)

    predictions={
            "classes":tf.argmax(input=logits,axis=1),
            "probabilities":tf.nn.softmax(logits,name="softmax_tensor")
            }
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
    loss=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
    tf.summary.scalar('loss',loss)
    if mode==tf.estimator.ModeKeys.TRAIN:
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op=optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)

    eval_metric_ops={
            "accuracy":tf.metrics.accuracy(
                labels=labels,predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
            mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)

def Scan(rootdir):
	img=[]
	listf=os.listdir(rootdir)
	for i in range(len(listf)):
		path=os.path.join(rootdir,listf[i])
		if os.path.isdir(path):
			img.extend(Scan(path))
		if os.path.isfile(path):
			img.append(path)
	return img

def readimg(imglist):
	height=28
	width=28
	num_images=len(imglist)
	data=np.zeros(shape=(num_images,height*width),dtype=np.int64)
	for i in range(num_images):
		img=cv2.imread(imglist[i],cv2.IMREAD_GRAYSCALE)
		if img is None:
			print(imglist[i])
			exit(0)
		img=cv2.resize(img,(height,width))
		img=img.reshape([height*width])
		data[i,:]=img

	data=data/255.0

	return data


if __name__=="__main__":
#img=cv2.imread(imgpath)
#cv2.imshow("test",img)
	imglist=Scan(imgpath)
	test_data=readimg(imglist)
	atom_classifier=tf.estimator.Estimator(
			model_fn=cnn_model_fn,
			model_dir="models/atom_classify_sixtypes_model")
	predict_input_fn=tf.estimator.inputs.numpy_input_fn(
			x={"x":test_data},
			num_epochs=1,
			shuffle=False)
	result=atom_classifier.predict(input_fn=predict_input_fn)
	atom_type={0:'原子间隙',1:'间隙（上下）',2:'间隙（左右）',3:'四分之一原子',4:'半原子',5:'完整原子'}

	for i, p in enumerate(result):
#print("Probabilities %s: %s" % (i+1,p["probabilities"]))
		print("Prediction %s: %s" % (i+1,atom_type[p["classes"]]))
