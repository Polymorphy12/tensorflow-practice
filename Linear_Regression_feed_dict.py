#Linear Regression으로 추측할 데이터를 미리 정해놓지 않고 내가 정의해서 주고 싶을 때.
#placeholder을 써야겠지.
import tensorflow as tf




#순서를 먼저 인지하자.
#1.그래프를 만든다.
#2.세션을 만들어서
#3.변수를 초기화 해주고
#4.세션을 이용해 optimizer들을 실행시킨다.

#rank 1인 random normal 변수를 만들어놓는다.
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#feed_dict,
#즉 feed dictionary를 이용해 데이터를 받을 텐서를 위해 placeholder을 사용한다.
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])


#가설함수와 cost 함수를 만들고
hypothesis = X*W +b

cost = tf.reduce_mean(tf.square(hypothesis -Y))

#Optimizer을 설정해 학습과정을  설정한다. (여기선 Gradient Descent를 이용해서 최소화 시킬거야.)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#여기까지 Data Flow Graph를 만들어놨어.
#이제 세션을 만들어서 Graph를 시작(Launch) 하자.
session = tf.Session()
#변수들 초기화 시켜주고
session.run(tf.global_variables_initializer())




for step in range(6000):
    #파이썬에서 백슬래쉬는 구문이 길어질 때, 다음줄까지 구문을 잇겠다는걸 의미한다.
    cost_val, W_val, b_val, _ = \
              session.run([cost,W,b,train],
                          feed_dict={X: [1,2,3,4], Y:[0,-1,-2,-3]})

    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)
    
