#Linear Regression with placeholder

#Linear Regression이 뭔지 이전 프로그램에서 알아봤으니 변수바꾸는 placeholder을 사용해보자.
#시작하기 전에 코드를 보자.




import tensorflow as tf

x_data = [1.,2.,3.]
y_data = [1.,2.,3.]



# 기울기 W와 y절편 b를 찾아서 y_data = W * x_data +b를 계산할거야.
# 솔직히 간단해서 우리가 바로 보고 알 수 있지만텐서플로우가 직접 찾게 만들자.


#random uniform의 뜻이 뭐더라?
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))



#float형 변수 X와 Y.

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)



#가설 함수. x_data를 바로 넣어주지 않은 것을 볼 수 있다.  Our Hypothesis 
#placeholder을 사용한 이유는 hypothesis 식을 새로만들 필요가 없기 때문에.
#자바 기초에서 변수가 왜 필요하다고 했지? 일일이 만들기 귀찮으니까.
hypothesis = W*X+b




#간단하게 만든 cost function x_data를 바로 넣어주지 않은 것을 볼 수 있다..
cost = tf.reduce_mean(tf.square(hypothesis - Y))


a = tf.Variable(0.1) #Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)



#시작하기전에 변수들을 모두 초기화 시킨다. 이 초기화 시키는 것도 'run' 해야한다.
#TF 1.0 미만 버전에서는 init = tf.initialize_all_variables()

init = tf.global_variables_initializer()


sess = tf.Session()
sess.run(init)


for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data}) #placeholder 부분에 x_data, y_data를 넣어줬다.
                                                    #함수를 실행시킬 때마다 feed_dict로 변수 넣어줘야한다.
    if step %20 == 0:
        print(step, sess.run(cost , feed_dict={X:x_data, Y:y_data}),sess.run(W),sess.run(b))
