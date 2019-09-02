import tensorflow as tf
import matplotlib.pyplot as plt


#tf 그래프 입력
X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_sample = len(X)

#Set model weights
W = tf.placeholder(tf.float32)


#Linear model을 만들자.
hypothesis = tf.multiply(X,W)


#cost function
cost = tf.reduce_sum(tf.pow(hypothesis-Y, 2))/(m)


#변수들을 모두 초기화해줄 필요가 있다. 그리고 이 녀석은 Session.run으로 실행시켜줄 필요가 있다.
init = tf.global_variables_initializer()

#그래프 그릴거야.

W_val = []
cost_val = []

#세션 만들고, 초기화를 실행해. tf.run()
sess = tf.Session()
sess.run(init)
for i in range(-30, 50):
    print(i*0.1, sess.run(cost, feed_dict={W : i * 0.1}))
    W_val.append(i*0.1)
    cost_val.append(sess.run(cost, feed_dict={W : i * 0.1}))



plt.plot(W_val, cost_val, 'ro')
plt.xlabel('Cost')
plt.ylabel('W')
plt.show()

