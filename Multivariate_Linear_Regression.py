import tensorflow as tf


x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142.]


#feed_dict로 넘겨주기 위해서 placeholder 만들어주고.
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)


#weight들과 bias를  random normal로 준다.

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')


#multivariate regression이므로 변수도 세개, weight도 세개를 준다.
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b
print(hypothesis)


#cost function을 만들어서 gradient descent해서 최솟값을 구해야겠지요?

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize. Need a very small learning rate for this data set (le
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)


#여기까지 graph 만들어놨고. 세션을 만들어서 실행해보자.
sess = tf.Session()
# variables 초기화 시켜주는건 말할 것도 없고.
sess.run(tf.global_variables_initializer())

##파이썬에서 언더 스코어(_)의 의미는 여러가지이다.
##
##1. 인터프리터(Interpreter)에서 마지막 값을 저장할 때
##
##2. 값을 무시하고 싶을 때 (흔히 “I don’t care”라고 부른다.)
##
##3. 변수나 함수명에 특별한 의미 또는 기능을 부여하고자 할 때
##
##4. 국제화(Internationalization, i18n)/지역화(Localization, l10n) 함수로써 사용할 때
##
##5. 숫자 리터럴값의 자릿수 구분을 위한 구분자로써 사용할 때
##여기서는 2번이겠지.

for step in range(2001):
    cost_val, hy_val, w_val1,w_val2,w_val3, _ = sess.run([cost, hypothesis, w1,w2,w3,  train],
                                   feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val,"\nWeight1: ", w_val1,
              "\nWeight2: ", w_val2,
              "\nWeight3: ", w_val3,"\nPrediction:\n", hy_val,"\n")

        
