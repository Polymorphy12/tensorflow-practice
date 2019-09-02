#자, Linear Regression을 시각화 해보자.
#시각화 할 떄는 matplotlib.pyplot을 사용할꺼야.
import tensorflow as tf
import matplotlib.pyplot as plt


X = [1,2,3]
Y = [2,4,6]

W = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)


hypothesis = X * W


cost = tf.reduce_mean(tf.square(hypothesis - Y))




sess = tf.Session()

#X축
W_history = []
#Y축
cost_history = []


for i in range(-30,50):
    curr_W = i * 0.1
    curr_cost = sess.run(cost, feed_dict = {W: curr_W})
    W_history.append(curr_W)
    cost_history.append(curr_cost)


plt.plot(W_history, cost_history)
plt.show()
