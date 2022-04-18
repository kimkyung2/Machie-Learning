# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 13:34:06 2022

@author: kki
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def logistic_calssification():
    x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
    y_data = [[0], [0], [0], [1], [1], [1]]
    
    x = tf.placeholder(tf.float32, shape = [None, 2])
    y = tf.placeholder(tf.float32, shape = [None, 1])
    
    W = tf.Variable(tf.random_normal([2, 1]), name = 'weight') # matrix(x * w), x가 2개 들어올 때, W가 2개가 됨, 나가는 값은 y 1개
    b = tf.Variable(tf.random_normal([1]), name ='bias') # 항상 나가는 값과 같음
    
    # 
    hypothesis = tf.sigmoid(tf.matmul(x, W) + b) # H(x) = 1+e의 -WTX승 분의 1
    
    cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis)) # cost(W)이고 tf.reduce_mean()은 - m분의 1 시그마
    
    train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost) # Gradient사용해서 W - cost(W) 미분하기
    
    predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32) # 0.5를 기준으로해서 0.5보다 크면 pass 작으면 fail, 이때 float형으로 바꾸면 True(pass) = 1, False = 0
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32)) # equal을 사용해서 예측값(predicted)와 실제값(y)의 값이 같으면 1, 아니면 0
    
    # 학습하는 모델 만들기
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for step in range(10001):
            cost_val, _ =sess.run([cost, train], feed_dict = {x :x_data, y: y_data})
            if step % 200 == 0:
                print(step, cost_val)
                
        h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {x: x_data, y: y_data})
        print('Hypothesis\n ', h, '\nCorrect (y): ', c, '\nAccuracy: ', a)
        
# logistic_calssification()

import tensorflow as tf

def tensorflow2_0():
    x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
    y_data = [[0], [0], [0], [1], [1], [1]]
    
    tf.model = tf.keras.Sequential()
    tf.model.add(tf.keras.layers.Dense(units = 1, input_dim=2))
    tf.model.add(tf.keras.layers.Activation('sigmoid'))
    
    tf.model.compile(loss='binary_crossentropy', optimizer = tf.keras.optimizers.SGD(lr = 0.01), metrics = ['accuracy'])
    tf.model.summary()
    
    history = tf.model.fit(x_data, y_data, epochs = 5000)
    
    print("Accuracy: ", history.history['accuracy'][-1])
        
tensorflow2_0()