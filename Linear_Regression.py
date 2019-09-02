#Linear Regression TensorFlow
import tensorflow as tf




# X와 Y데이터, 즉 X가 1일 때 Y가 1이 되고 X가 3일 때 Y가 3인 linear regression을 해봐라.
x_train = [1,2,3]
y_train = [1,2,3]





#y_data = x_data*W + b인 W와 b를 찾으려고 한다.
# 솔직히 딱봐도 W는 1이고 b는 0 이라는걸 잘 알겠지만
#tensorflow가 직접 찾아내도록 해보자.

#rank 1인 random normal 변수를 만들어놓는다.
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#이제 가설함수를 한번 만들어볼까?
hypothesis = W*x_train + b


#cost function을 만들어 보자.
#(가설함수 - 실제 값)들의 제곱합의 평균이었지?
cost = tf.reduce_mean(tf.square(hypothesis-y_train))


#이제 Gradient Descent 알고리즘을 진행할거야.
#텐서플로우가 참 편한게 optimizer가 함수로 구현되어 있다는거야.
#cost가 최솟값이 되도록 만들고 싶지.
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)


#여기까지 Data Flow Graph를 만들었어.
#이제 세션을 만들어보자. 세션은 함수들이 실제 동작하도록 해주는 역할을 하는 것 같아.
sess = tf.Session()
#Session에서 사용될 Variables는 사용되기 전에 초기화 시켜줘야 해.
sess.run(tf.global_variables_initializer())


#minimize 작업을 2001번 실행하겠다는 거야.
for step in range(4000):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

