# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:59:26 2022

@author: kki
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


###### 그래프 구현#####
# x, y 데이터 입력
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Variable : tf를 실행시키면 자체적으로 등록시키는 값
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1], name ='bias'))

hypothesis = x_train * W +b

# cost/loss function
# reduce_mean은 값을 집어넣었을 때 그 값의 평균을 계산해줌
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#################################

# variable을 사용하기 위해서는 반드시 global_variables_initializer을 실행해야함
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 ==0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
        
#### Placeholders을 사용해서 만들기####
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, Shape=[None])

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                         feed_dict = {x:[1, 2, 3], y:[1, 2, 3]})
        
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

