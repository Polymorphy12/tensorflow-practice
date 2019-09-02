#PlaceHolder 연습.
import tensorflow as tf


#placeholder로 변수타입을 설정한다.
#변수 타입이 int16이다.
a = tf.placeholder(tf.int16)

b = tf.placeholder(tf.int16)



#연산자를 정의해보자.
add =  tf.add(a,b)

#Multiplication operator.
#For TF lower than 1.0, tf.multiply(a,b)
mul = tf.multiply(a,b)


with tf.Session() as sess:
    #Run every operation with variable input
    print("변수들의 덧셈 : %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("변수들의 곱셈 : %i" % sess.run(mul, feed_dict={a: 2, b: 3}))


    
