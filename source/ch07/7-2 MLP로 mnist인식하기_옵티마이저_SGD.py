import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# 1. MNIST 데이터셋
(x_train,y_train),(x_test,y_test)=ds.mnist.load_data()

# 2. 데이터 전처리
x_train=x_train.reshape(60000,784)  # 28 x 28 --> 784
x_test=x_test.reshape(10000,784)
x_train=x_train.astype(np.float32)/255.0   # 픽셀 값을 0 ~ 1 정규화
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

# 3. 신경망 구성
mlp=Sequential()
mlp.add(Dense(units=512,activation='tanh',input_shape=(784,)))  # 은닉층 512 , 입력층 784
mlp.add(Dense(units=10,activation='softmax'))   # 출력층 10

# 4. 모델 구조 출력
mlp.summary()

# 5. 모델 컴파일 (환경 설정)
mlp.compile(loss='MSE',optimizer=SGD(learning_rate=0.01),metrics=['accuracy'])  # SGD 확률적 경사 하강법

# 6. 모델 학습  (배치 size 128 --> 한번 128개 사용 )
mlp.fit(x_train,y_train,batch_size=128,epochs=50,validation_data=(x_test,y_test),verbose=2)

# 7. 최종 테스트 평가
res=mlp.evaluate(x_test,y_test,verbose=0)

print(f"테스트 손실: {res[0]:.4f}")
print(f"테스트 정확도: {res[1] * 100:.2f}%")