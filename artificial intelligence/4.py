import numpy as np #넘파이를 이용
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

import matplotlib.pyplot as plt #산점도
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
train_input = train_input.reshape(-1,1) #reshape() 메서드는 배열의 크기를 지정할 수 있다. -1은 나머지 원소 개수로 모두 채우라는 의미
test_input = test_input.reshape(-1,1)  #사이킷런을 사용하기 위해서 2차원 배열로 바꾸어줌
#print(train_input.shape, test_input.shape) 배열 형태 확인
#사이킷런에서 k-최근접 이웃 회귀 알고리즘을 구현한 클래스는 KNeighborsRegressor이다. 사용법은 KNeighborsClassifier과 유사
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor() #객체 생성
knr.fit(train_input, train_target) #k-최근접 이웃 회귀 모델을 훈련
#print(knr.score(test_input, test_target)) 결정계수 R^2라고도 함
from sklearn.metrics import mean_absolute_error #sklearn.metrics패키지 아래 mean_absolute_error은 타깃과 예측의 절댓값 오차를 평균하여 반환한다.
test_prediction = knr.predict(test_input) #테스트 세트에 대한 예측을 만듬
#mae = mean_absolute_error(test_target, test_prediction) 테스트 세트에 대한 평균 절댓값 오차를 계산
#print(mae) 예측이 평균적으로 19g 정도 타깃값과 다르다는 것을 알 수 있다.

#훈련 세트와 테스트 세트로 결정계수를 확인하여 훈련 세트>>>>>테스트 세트 이면 훈련 세트에 과대적합(훈련 세트에만 잘 맞는 모델이라 테스트 세트와 나중에 실전에서 사용하기 어려움)
                                              #훈련 세트<<<<<테스트 세트(또는 둘다 낮음) 이면 훈련 세트에 과소적합(모델이 너무 단순하여 훈련 세트에 적절히 훈련되지 않음)
#print(knr.score(train_input, train_target)) 훈련 세트의 결정계수 < 테스트 세트의 결정계수 이므로 과소적합이다.  과소적합을 해결하기 위해선 모델을 복잡하게 만들어야 하는데 k-최근접 이웃 알고리즘에서는 이웃의 개수를 줄이면 됌
knr.n_neighbors = 3
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target)) 
print(knr.score(test_input, test_target)) #k의 개수를 줄여줌으로써 과소적합을 해결함 또한 두 결정계수의 차이가 크지않으므로 과대적합이 된 것도 아님

#과대적합과 과소적합에 대한 이해를 위한 복잡한 모델과 단순한 모델
knr = KNeighborsRegressor() #k-최근접 이웃 회귀 객체를 만듬
x = np.arange(5, 45).reshape(-1, 1) #5에서 45까지 x좌표를 만든 후 2차원 배열로 만듬
for n in [1,5,10]:#n이 1,5,10일 때 예측 결과를 그래프로 그림
  knr.n_neighbors = n
  knr.fit(train_input, train_target) #모델 훈련
  prediction = knr.predict(x) #지정한 범위 x에 대한 예측을 구함
  plt.scatter(train_input, train_target) #훈련 세트와 예측 결과를 그래프로 그림
  plt.plot(x, prediction)
  plt.title('n_neighbors = {}'.format(n))
  plt.xlabel('length')
  plt.ylabel('weight')
  plt.show()  #결과적으로 k이웃값이 늘어날수록 단순해진다.
