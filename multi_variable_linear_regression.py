# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:56:01 2022

@author: kki
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def multi_variable():
    # 데이터 입력
    x1_data = [73., 93., 89., 96., 73.]
    x2_data = [80., 88., 91., 98., 66.]
    x3_data = [75., 93., 90., 100., 70.]
    y_data = [152., 185., 180., 196., 142.]
    
    # placeholders 
    x1 = tf.placeholder(tf.float32)
    x2 = tf.placeholder(tf.float32)
    x3 = tf.placeholder(tf.float32)
    
    y = tf.placeholder(tf.float32)
    
    w1 = tf.Variable(tf.random_normal([1]), name='weight1')
    w2 = tf.Variable(tf.random_normal([1]), name='weight2')
    w3 = tf.Variable(tf.random_normal([1]), name='weight3')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    
    hypothesis = x1*w1 + x2*w2 + x3*w3 + b
    
    
    # cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    #optimmizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
    train = optimizer.minimize(cost)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                       feed_dict={x1: x1_data, x2:x2_data, x3:x3_data, y:y_data})
        if step % 10 ==0:
            print(step, "Cost: ", cost_val, "Prediction: ", hy_val)
            
# multi_variable()

# 데이터의 길이가 n개이면 none을 줘서 만들 수 있음

def multi_variable2():
    x_data = [[73., 80., 75.], [93., 88., 93.],
              [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
    y_data = [[152.], [185.], [180.], [196.], [142.]]
    
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([3, 1]), name = 'weight')
    b = tf.Variable(tf.random_normal([1]), name = 'bias')
    
    hypothesis = tf.matmul(x, W) + b
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
    train = optimizer.minimize(cost)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x:x_data, y:y_data})
        if step % 10 ==0:
            print(step, "cost: ", cost_val, "Prediction: ", hy_val)
    

multi_variable2()



















