# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 18:21:13 2022

@author: kki
"""

import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def costfuncion():
    
    # 데이터 입력
    x = [1, 2, 3]
    y = [1, 2, 3]
    
    # 가설로 만든 linear model = x*W
    W = tf.placeholder(tf.float32)
    hypothesis = x * W
    
    # cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    
    # graph 고정하기위함
    sess = tf.Session()
    # variable을 초기화
    sess.run(tf.global_variables_initializer())
    
    # 그래프를 고정시키기 위한 준비 cost function variables
    W_val = []
    cost_val = []
    
    for i in range(-30, 50):
        feed_W = i *0.1
        curr_cost, curr_W = sess.run([cost, W], feed_dict = {W: feed_W})
        W_val.append(curr_W)
        cost_val.append(curr_cost)
        
    plt.plot(W_val, cost_val)
    plt.show()

# costfunction()


### 미분해서 직접 minimizing cost 구하기
def costfunction2():
    
    # 데이터 입력
    x_data = [1, 2, 3]
    y_data = [1, 2, 3]
    
    # 가설로 만든 linear model = x*W
    W = tf.Variable(tf.random_normal([1]), name = 'weight')
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    
    hypothesis = x * W
    
    # cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    
    # Gradient descent
    # minimize : Gradient Descent using derivative : W -= learning_rate * derivative
    learning_rate = 0.1 # 알파 값
    gradient = tf.reduce_mean((W*x - y)*x) # w의 식을 미분한 값
    descent = W - learning_rate * gradient
    update = W.assign(descent) # assign 함수를 사용해서 upgrade
    
    sess = tf.Session()
    
    sess.run(tf.global_variables_initializer())
    for step in range(21):
        sess.run(update, feed_dict = {x: x_data, y: y_data})
        print(step, sess.run(cost, feed_dict={x : x_data, y: y_data}), sess.run(W))
        
costfunction2()

#### gradientDescentOptimizer 사용해서 최소값 cost 구하기
def costfunction3():
    x = [1, 2, 3]
    y = [1, 2, 3]
    
    W = tf.Variable(5.0)
    hypothesis = x * W
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
    train = optimizer.minimize(cost)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for step in range(100):
        print(step, sess.run(W))
        sess.run(train)

costfunction3()

# Gradient 응용으로 만든 식과 손으로 미분해서 만든 식과의 결과 비교
def costfunction4():        
    x = [1, 2, 3]
    y = [1, 2, 3]
    W = tf.Variable(5.0)
    hypothesis = x * W
    gradient = tf.reduce_mean((W*x-y)*x)*2
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    
    gvs = optimizer.compute_gradients(cost,[W])
    apply_gradients = optimizer.apply_gradients(gvs)
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for step in range(100):
        print(step, sess.run([gradient, W, gvs]))
        sess.run(apply_gradients)

costfunction4()

    