#Linear Regression이 뭔지 알아보는 프로그램.




import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]



# 기울기 W와 y절편 b를 찾아서 y_data = W * x_data +b를 계산할거야.
# 솔직히 간단해서 우리가 바로 보고 알 수 있지만텐서플로우가 직접 찾게 만들자.


#random uniform의 뜻이 뭐더라?
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))



#가설 함수.  Our Hypothesis
hypothesis = W*x_data +b



#간단하게 만든 cost function.
cost = tf.reduce_mean(tf.square(hypothesis - y_data))


a = tf.Variable(0.1) #Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)



#시작하기전에 변수들을 모두 초기화 시킨다. 이 초기화 시키는 것도 'run' 해야한다.
#TF 1.0 미만 버전에서는 init = tf.initialize_all_variables()

init = tf.global_variables_initializer()


sess = tf.Session()
sess.run(init)


for step in range(2001):
    sess.run(train)
    if step %20 == 0:
        print(step, sess.run(cost),sess.run(W),sess.run(b))
