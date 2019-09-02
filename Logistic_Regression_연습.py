#Logistic Regression 텐서플로우로 구현하기.

import tensorflow as tf


#독립변수들과, 그 독립변수에 따라 정해지는 종속변수가 주어져있다.

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]

y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]


#데이터를 받을 place holders를 정의해줘야 하지.

#n*2 행렬
X = tf.placeholder(tf.float32, shape=[None, 2]) #한 행에 두개의 열이 들어가고, 몇개의 행이든 들어갈 수 있다.
#n*1 행렬
Y = tf.placeholder(tf.float32, shape=[None, 1]) #한 행에 하나의 열이 들어가고, 몇개의 행이든 넣을 수 있다.


W = tf.Variable(tf.random_normal([2, 1]), name='weight') #2*1행렬.
b = tf.Variable(tf.random_normal([1]), name='bias')



#시그모이드 함수를  활용한 가설함수.
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)


#cost function
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))


#학습시킬거야. 시그모이드 cost function은 convex function이므로
#gradient descent로 최솟값을 구할 수 있어.
#task를 정의하자.
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)


# Accuracy computation
# True if hypothesis > 0.5

#predicted는 hypothesis를 이용해서 얻은 예측값이야.
#이제 predict값을 구할거야.
#
#cast 함수는 형변환을 시켜주는 함수야.
#
#첫번째 매개변수는 참, 거짓 중 하나의 값을 가지는 조건연산이지.
#
#여기서 사용된 조건 연산은 hypothesis가 실제 값을 맞췄는지 확인하는 연산이야.
#
#참의 값은 1이고 거짓의 값은 0이지.
#
# hypothesis > 0.5 면 predicted의 값을 float32형의 1로, 아니면 0으로 정하겠다는 뜻이야.
#
predicted = tf.cast(hypothesis >0.5, dtype = tf.float32)

#얼마나 정확한지 확률을 구할 수 있어.
#
#예측값과 실제 Y값이 같은지 확인할 수 있겠지.
#
#elementwise로 비교한 참 거짓값(0 또는 1이겠지.)의 평균을 구하면 정확도를 구할 수 있어.
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))





#그래프를 실행해보자.
#for문의 각 단계마다 현재까지 지나온 학습단계 수와 cost 값과 W 값과 b값을 출력하고 학습을 진행할거야. 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        #파이썬의 언더바 '_'는 변수를 무시할 수 있는 기능이 있어.
        cost_val, w_val, b_val, _ = sess.run([cost, W, b, train], feed_dict = {X: x_data, Y: y_data})
        #다 출력하면 귀찮으니까 200번에 한번씩만 출력하도록 만들자.
        if step %200 == 0 :
            print("Step = ",step, "\nCost = \n",cost_val,"\nW = \n",w_val, "\nb = \n",b_val)


    #얼마나 예측값이 정확한지 리포트를 작성해보자.
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})

    print("\nHypothesis: \n", h, "\nCorrect (Y): \n", c, "\nAccuracy: ", a)
