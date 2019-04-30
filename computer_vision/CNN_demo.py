# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:23:06 2019

@author: freej
"""
#input
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
 
from tensorflow.examples.tutorials.mnist import input_data
#one_hot=True 表示对label进行one-hot编码，比如标签4可表示为[0,0,0,0,1,0,0,0,0,0,0]
mnist = input_data.read_data_sets('data/',one_hot=True)

#2.设置参数
#tf.reset_default_graph()函数:
#用于清除默认图形堆栈并重置全局默认图形.默认图形是当前线程的一个属性。该tf.reset_default_graph函数只适用于当前线程。
#当一个tf.Session或者tf.InteractiveSession激活时调用这个函数会导致未定义的行为。
#调用此函数后使用任何以前创建的tf.Operation或tf.Tensor对象将导致未定义的行为。

tf.reset_default_graph()
sess = tf.InteractiveSession()
x = tf.placeholder("float",shape = [None,28,28,1])   #28*28*1 不是784个像素点
y_ = tf.placeholder("float",shape = [None,10])

#3.第一层卷积层
#定义卷积核  5*5的卷积核 1维的原始输入图像 输出32通道 32种不同的核进行卷积得到32个不同的特征
W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1))
#偏置参数 用任意一个常数进行初始化
b_conv1 = tf.Variable(tf.constant(.1,shape = [32]))
#filter：卷积核 strides:步长 4个1 batchsize,h滑多少 ,w滑多少 ,通道数   SAME：可以自动补充
h_conv1 = tf.nn.conv2d(input = x,filter = W_conv1,strides = [1,1,1,1],padding = 'SAME') + b_conv1
#卷积一次 进行一次映射
h_conv1 = tf.nn.relu(h_conv1)  
#池化层 不会改变特征的个数
#没有参数 ksize:batchsize,2*2区域,通道数 strides:步长 4个1 batchsize,h滑多少 ,w滑多少 ,通道数
h_pool1 = tf.nn.max_pool(h_conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

#4.第二个卷积层
# 第二个卷积层 也可以写成函数
def conv2d(x,W):
	return tf.nn.conv2d(input = x, filter = W, strides = [1,1,1,1],padding = 'SAME')
def max_pool_2x2(x):	
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
#第二个卷积核
W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(.1,shape = [64]))
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#5.全连接层：矩阵相连
W_fcl = tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))	#7*7*64转换为1024个特征
b_fcl = tf.Variable(tf.constant(.1,shape = [1024]))
#reshape将7*7*64拉伸成一长条,即h_pool2   -1自动取了一个合适的值，1
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fcl = tf.nn.relu(tf.matmul(h_pool2_flat,W_fcl) + b_fcl)


#防止过拟合
#droput layer:杀死一部分神经元 防止过拟合
#指定保留率
keep_prob = tf.placeholder("float")
h_fcl_drop = tf.nn.dropout(h_fcl,keep_prob)

#6.第二个全连接层
W_fc2 = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))	#
b_fc2 = tf.Variable(tf.constant(.1,shape = [10]))

#7.最后一层
y = tf.matmul(h_fcl_drop,W_fc2) + b_fc2

#8. 求损失值，指定优化器
#用损失函数求交叉熵的损失
crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_,logits = y))
#指定优化器 AdamOptimizer可以自适应的调整学习率，比梯度下降好
trainStep = tf.train.AdamOptimizer().minimize(crossEntropyLoss)
#计算正确率
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))


#9.开始训练
sess.run(tf.global_variables_initializer())
batchsize = 50
for i in range(1000):
	batch = mnist.train.next_batch(batchsize)
	trainingInputs = batch[0].reshape([batchsize,28,28,1])		#输入大小必须为28*28*1
	trainingLabels = batch[1]
	if i%100 == 0:
		trainAccuracy  = accuracy.eval(session = sess,feed_dict = {x:trainingInputs,y_:trainingLabels,keep_prob:0.5})
		print("step %d, training accuracu %g"%(i,trainAccuracy))
	trainStep.run(session = sess,feed_dict={x:trainingInputs,y_:trainingLabels,keep_prob:0.5})
    



