#Logistic Regression으로 당뇨병 예측하기.

import tensorflow as tf
import numpy as np


#csv 파일을 옮겨 오는거야.
#numpy의 indexing 기능을 사용해서  독립변수와 종속변수를 구분해보자.
#맨마지막 열을 제외한 열들은 독립변수들,
#그리고 맨 마지막 열은 종속 변수로 사용하는거야.
xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


#X 데이터의 모양과 Y데이터의 모양을 출력해보자
#placeholder의 shape를 예측하려면 각 데이터의 열의 개수를 알 필요가 있어. 
print(x_data.shape, y_data.shape)


#이 윗줄까지 실행해보니까 독립변수들의 종류는 8개야. 
#x_data와 y_data를 담을 placeholder를 정의해놓자.
X = tf.placeholder(tf.float32, shape = [None, 8])
Y = tf.placeholder(tf.float32, shape = [None, 1])


W = tf.Variable(tf.random_normal([8,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

#가설함수를 설정해주고
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)


#cost 함수를 설정해주는거야.
cost = -tf.reduce_mean(Y * tf.log(hypothesis)+ (1-Y) * tf.log(1-hypothesis))

#이제 cost값이 최소가 되도록 Gradient descent로 학습을 시켜주는거야.
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

#----------------------여기까지 실행시킬 그래프 하나.--------------------------

#그래서 이제 종속변수 예측값을 구하고 정확도를 구할거야.
predicted = tf.cast(hypothesis >=0.5, tf.float32)

accuracy = tf.reduce_mean( tf.cast(tf.equal(predicted,Y), tf.float32) )

#----------------------여기까지 실행시킬 그래프 둘.--------------------------



#여기까지 그래프를 정의했고, 그래프를 실행해보자.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #첫번째 그래프를 실행하자.
    for steps in range(30001):
        #학습을 시켜주면서 알고싶은 값도 같이 꺼내놓으면 좋을것 같애.
        #각 학습 단계마다 cost값은 얼마인지, 가중치(cost)값은 얼마인지, bias 값은 얼마인지
        #기록해두자.
        cost_val, w_val, bias, _ = sess.run([cost, W, b, train], feed_dict= {X: x_data, Y: y_data})
        if steps %600 == 0 :
            print("\nstep : \n", steps,"\nCost : \n", cost_val,"\nWeight : \n", w_val,"\nBias : \n", bias)


    #두번째 그래프를 실행하자.
    #최종적으로 학습된 가설함수에 데이터를 주었을 때 종속변수 예측값과 정확도를 알고싶어.
    h,predicted_val,accuracy_val = sess.run([hypothesis, predicted, accuracy], feed_dict = {X: x_data, Y: y_data})
    print("\nhypothesis : \n", h,"\npredicted_val : \n", predicted_val,"\accuracy : \n", accuracy_val)
