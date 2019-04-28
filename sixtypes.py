from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import cv2
import random

tf.logging.set_verbosity(tf.logging.INFO)

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

def load(imglist,labels_value):
    num_images=len(imglist)
    height=28
    width=28
    channels=1

    #data=np.zeros(shape=(num_images,height,width,channels),dtype=np.int64)
    #labels=np.zeros(shape=(num_images,1),dtype=np.int32)
    data=np.zeros(shape=(num_images,height*width),dtype=np.int64)
    labels=np.zeros(shape=(num_images,1),dtype=np.int32)

    for i in range(num_images):
        img=cv2.imread(imglist[i],cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(imglist[i])
            exit(0)

        img=cv2.resize(img,(height,width))
        #img=img.reshape([height,width,channels])
        img=img.reshape([height*width])
        #data[i,:,:,:]=img
        data[i,:]=img
        labels[i,:]=labels_value[i]

    data=data/255.0

    return data,labels


def Inputs(imglist,flags):

    labels=[]
    num=len(imglist)
    for i in range(num):
        labels.append(flags)
    data=[imglist,labels]
    return data
    #data=np.array(data)
    #data=data.transpose()
    #imgpath=data[:,0]
    #labels=data[:,1]
    #return imgpath,labels
"""
    positive=np.hstack([imglist_P,1])
    negative=np.hstack([imglist_N,0])
    data=np.vstack([positive,negative])
    np.random.shuffle(data)
    data=data[:,:-1]
    labels=data[:,-1]
    return data
"""

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


    img=[]
    listf=os.listdir(rootdir)
    for i in range(len(listf)):
        path=os.path.join(rootdir,listf[i])
        if os.path.isdir(path):
            img.extend(Scan(path))
        if os.path.isfile(path):
            img.append(path)
    return img

def ImgPath(folderlist,labelslist):
    path=[]
    for i in range(len(folderlist)):
        abspath=Scan(folderlist[i])#image folder 'AI','blank' et_al
        data_i=Inputs(abspath,labelslist[i])#data_i is list[abspath,labels]
        path.append(data_i)#path[[[AI/*.jpg...],[0..0]],[[digits/*.jpg...],[1...1]],[[blank/*.jpg...],[2..2]]]

    X=[]
    Y=[]
    for j in path:
        X.extend(j[0])#j[0] AI/*.jpg digits/*.jpg
        Y.extend(j[1])#j[1] 0...0 1...1
    data=[X,Y]
    data=np.array([X,Y])
    data=np.transpose(data)
    np.random.shuffle(data)
    row,col=data.shape
    train_path=data[0:int(0.7*row),0]
    eval_path=data[int(0.7*row):,0]

    train_values=data[0:int(0.7*row):,1]
    eval_values=data[int(0.7*row):,1]
    return (train_path,train_values),(eval_path,eval_values) 

if __name__=="__main__":
    folderlist=['nonatom/null/','nonatom/gap/','nonatom/interval/','nonatom/quarter/','nonatom/half/','train_atom/atom/']
    labelslist=[0,1,2,3,4,5]

    (train_path,train_values),(eval_path,eval_values)=ImgPath(folderlist,labelslist)

    train_data,train_labels=load(train_path,train_values)
    eval_data,eval_labels=load(eval_path,eval_values)
    print(train_data.shape,train_labels.shape)
    print(eval_data.shape,eval_labels.shape)
    atom_classifier=tf.estimator.Estimator(
            model_fn=cnn_model_fn,model_dir="models/atom_classify_sixtypes_model")

    tensors_to_log={"probabilities":"softmax_tensor"}
    logging_hook=tf.train.LoggingTensorHook(
            tensors=tensors_to_log,every_n_iter=50)

    train_input_fn=tf.estimator.inputs.numpy_input_fn(
            x={"x":train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
    atom_classifier.train(
            input_fn=train_input_fn,
            steps=20000,
            hooks=[logging_hook])

    eval_input_fn=tf.estimator.inputs.numpy_input_fn(
            x={"x":eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False
            )
    eval_results=atom_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    
