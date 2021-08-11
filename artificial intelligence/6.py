#복잡하고 많은 특성을 이용해 다중 회귀를 하기위해 pandas를 사용함

import pandas as pd # pd는 관례적으로 사용하는 판다스의 별칭
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
#print(perch_full)
import numpy as np
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)
#사이킷런은 특성을 만들거나 전처리하기 위한 다양한 클래스를 제공한다. 사이킷런에서는 이런 클래스를 변환기라고 부른다.
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(include_bias = False) #사이킷런에서는 굳이 include_bias = False 할 필요가 없다
poly.fit(train_input)
train_poly = poly.transform(train_input)
#print(train_poly.shape)
#poly.get_feature_names()를 사용하면 각 특성이 어떻게 만들어졌는지 확인할 수 있다.
test_poly = poly.transform(test_input) #훈련 세트와 테스트 세트 모두 변환 해야 한다.
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
#print(lr.score(train_poly, train_target)) 0.9903183436982124
#print(lr.score(test_poly, test_target)) 0.9714559911594132
poly = PolynomialFeatures(degree = 5, include_bias = False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
#print(train_poly.shape) (42, 55) 특성의 개수가 55개
lr.fit(train_poly, train_target)
#print(lr.score(train_poly, train_target)) 0.9999999999991096
#print(lr.score(test_poly, test_target)) -144.40579242335605  훈련 세트에 과대적합 되어있어 테스트 세트에는 엄청 큰 음수가 나오게 된다. 이런 문제를 해결하기 위해선 특성을 줄여야한다.


#규제는 머신러닝 모델이 훈련 세트를 너무 과도하게 학습하지 못하도록 훼방하는 것을 말한다. 즉 모델이 훈련 세트에 과대적합되지 않도록 만드는 것
#선형 회귀 모델의 경우 특성에 곱해지는 계수(또는 기울기)의 크기를 작게 만드는 일
#특성의 스케일이 정규화되지 않으면 여기에 곱해지는 계수 값도 차이 나게 되므로 일반적으로 선형 회귀 모델에 규제를 적용할 때 계수값의 크기가 서로 많이 다르면 공정하게 제어되지 않는다.
#그래서 규제를 적용하기 전에 먼저 정규화를 해야한다.
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly) #표준점수로 변환한 train_scaled와 test_scaled  훈련 세트에서 학습한 평균과 표준편차는 StandardScaler 클라스 객체의 mean_ scale_속성에 저장된다. 특성마다 계산하므로 55개의 평균과 표준 편차가 들어있다.

#선형 회귀 모델에 규제를 추가한 모델을 릿지와 라쏘라고 부른다. 두 모델을 규제를 가하는 방법이 다르다.
#릿지는 계수를 제곱한 값을 기준으로 규제를 적용한다.
#라쏘는 계수의 절댓값을 기준으로 규제를 적용한다. 일반적으로 릿지를 더 선호

#릿지 회귀
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
#print(ridge.score(train_scaled, train_target)) 0.9896101671037343
#print(ridge.score(test_scaled, test_target)) 0.9790693977615396  테스트 세트와 훈련 세트의 점수가 정상으로 돌아왔다.
#릿지와 라쏘 모델을 사용할 때 규제의 양을 임의로 조절할 수 있다. 모델 객체를 만들 때 alpha 매개변수로 규제의 강도를 조절한다.
#alpha 값이 크면 규제 강도가 세지므로 계수 값을 더 줄이고 조금 더 과소적합되도록 유도한다.
#alpha 값이 작으면 규제 강도가 약해지므로 걔수 값을 줄이는 역할이 줄어들고 선형 회귀 모델과 유사해지므로 과대적합될 가능성이 크다.
#적절한 alpha 값을 찾는 한 가지 방법은 alpha 값에 대한 R^2 값의 그래프를 그려 보는 것이다. 훈련 세트와 테스트 세트의 점수가 가장 가까운 지점이 최적의 alpha값이 된다.
import matplotlib.pyplot as plt
train_score = []
test_score = []
#다음은 alpha값을 0.001에서 100까지 10배씩 늘려가며 릿지 회귀 모델을 훈련한 다음 훈련ㅅ ㅔ트와 테스트 세트의 점수를 파이썬 리스트에 저장한다.
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
  ridge = Ridge(alpha = alpha)
  ridge.fit(train_scaled, train_target)
  train_score.append(ridge.score(train_scaled, train_target))
  test_score.append(ridge.score(test_scaled, test_target)) #테스트 세트 점수와 훈련 세트 점수를 저장한다.
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
#그래프에서 보면 alpha값이 가장 가까운 곳이 0.1이므로 alpha값을 0.1로 하여 최종 모델을 훈련합니다.
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
#print(ridge.score(train_scaled, train_target)) 0.9903815817570366
#print(ridge.score(test_scaled, test_target)) 0.9827976465386916

#라쏘 회귀
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
#print(lasso.score(train_scaled, train_target)) 0.9897898972080961
#print(lasso.score(test_scaled, test_target)) 0.9800593698421883
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
  lasso = Lasso(alpha=alpha, max_iter=10000)
  lasso.fit(train_scaled, train_target)
  train_score.append(lasso.score(train_scaled, train_target))
  test_score.append(lasso.score(test_scaled, test_target)) 
  #라쏘 모델을 훈련할 때 ConvergenceWarning이란 경고가 발생할 수 있다. 사이킷런의 라쏘 모델은 최적의 계수를 찾기 위해 반복적인 계산을 수행하는데, 지정한 반복 횟수가 부족할 때 이런 경고가 발생한다.
  #이 반복 횟수를 충분히 늘리기 위해 max_iter 매개변수의 값을 10000으로 지정했다.
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
#그래프에서 보면 alpha값이 가장 가까운 곳이 1이므로 alpha값을 1로 하여 최종 모델을 훈련합니다.
lasso = Lasso(alpha = 1)
lasso.fit(train_scaled, train_target)
#print(lasso.score(train_scaled, train_target)) 0.9897898972080961
#print(lasso.score(test_scaled, test_target))   0.9800593698421883
#라쏘 모델은 계수 값을 아예 0으로 만들 수 있다. 라쏘 모델의 계수는 coef_속성에 저장되어 있다.
#print(np.sum(lasso.coef_ == 0)) 계수가 0인것이 42개나 있다.
