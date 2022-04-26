# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:48:22 2022

@author: kki
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def xor_logistic():
    x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = np.float32)
    y_data = np.array([[0], [1], [1], [0]], dtype = np.float32)
    
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    W = tf.Variable(tf.random_normal([2, 1]), name = 'weight')
    b = tf.Variable(tf.random_normal([1]), name = 'bias')
    
    hypothesis = tf.sigmoid(tf.matmul(x, W) + b)
    
    cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 -y) * tf.log(1 - hypothesis))
    train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)
    
    predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for step in range(10001):
            sess.run(train, feed_dict={x: x_data, y: y_data})
            if step % 100 == 0:
                print(step, sess.run(cost, feed_dict = {x:x_data , y:y_data}), sess.run(W))
                
        h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {x: x_data, y:y_data})
        print("\nHypothesis: ", h,"\nCorrect: ", c, "\nAccuracy: ",a)

# xor_logistic()

def xor_Nerual_network():
    x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = np.float32)
    y_data = np.array([[0], [1], [1], [0]], dtype = np.float32)
    
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    
    W1 = tf.Variable(tf.random_normal([2, 2]), name = 'weight1')
    b1 = tf.Variable(tf.random_normal([2]), name = 'bias1')
    layer1 = tf.sigmoid(tf.matmul(x, W1) + b1)
    
    W2 = tf.Variable(tf.random_normal([2,1]), name ='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name = 'bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) +b2)
    
    cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 -y) * tf.log(1 - hypothesis))
    train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)
    
    predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for step in range(10001):
            sess.run(train, feed_dict={x: x_data, y: y_data})
            if step % 100 == 0:
                print(step, sess.run(cost, feed_dict = {x:x_data , y:y_data}), sess.run([W1, W2]))
                
        h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {x: x_data, y:y_data})
        print("\nHypothesis: ", h,"\nCorrect: ", c, "\nAccuracy: ",a)
xor_Nerual_network()




















 