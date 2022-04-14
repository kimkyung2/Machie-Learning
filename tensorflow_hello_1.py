# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:43:11 2022

@author: kki
"""

## tf 2.0부터는 Session() 모듈을 지원하지않아 추가해야하는 내용##
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#################################################

hello = tf.constant("Hello, Tensorflow") # hello라는 노드를 생성

sess = tf.Session()

# print(sess.run(hello))

# 1) build graph
node1 = tf.constant(3.0, tf.float32) # 3 값을 가진 node를 생성
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2) # node3 = node1 + node2

print("node1:", node1, "node2:", node2) # 출력은 그래프의 하나의 요소다 라고 나옴, 결과값은 안나옴
print("node3:", node3)

# 2) sees.run(op)
sess = tf.Session() # 세션을 만듦
print("sess.run(node1, node2):", sess.run([node1, node2])) # 세션에 노드추가해서 결과값을 봄 
print("sess.run(node3):", sess.run(node3))


# Placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b

print(sess.run(adder_node, feed_dict = {a:3, b:4.5})) # feed_dict로 값을 adder_node로 넘겨줌
print(sess.run(adder_node, feed_dict = {a:[1,3], b:[2,4]}))

