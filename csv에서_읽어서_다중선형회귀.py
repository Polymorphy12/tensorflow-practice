#file에서 가져와서 multivariate linear regression  사용하기
import tensorflow as tf
import numpy as np


##python의 slicing 기능에 대해서 알 필요가 있다.
##
##그리고 numpy에서 좀 더 강력한 indexing, slicing 기능이 있는걸 알 필요가 있다.
##
##예를들어
##
##a = np.array([[1,2,3],[4,5,6]])이 있다고 하자.
##
##그렇다면 a[:, 0:-1]은  [[1,2],[4,5]]가 나온다.
##
##또한,
##
##a[:, [-1]]은   [[3],[6]]이 나오게 된다.

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Make sure the shape and data are OK
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)



#여기까지 실행되면 가설함수의 cost가 최소가 되는 최적의 weight들과 bias가 정해져 있다.
#따라서 지금부터 가설함수에 데이터를 넣었을 때 나오는 결과값은 최적의 예측값일 것이다.
#내 점수와 다른 사람들의 점수를 예측해보자.

# Ask my score
print("Your score will be ", sess.run(
    hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ", sess.run(hypothesis,
                                        feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
