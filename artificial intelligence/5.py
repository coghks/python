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
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors = 3)
knr.fit(train_input, train_target) # 최근접 이웃 회귀 모델을 훈련

#print(knr.predict([[50]])) 길이 50 농어의 무게를 예측했더니 [1033.33333333]가 나온다 하지만 실제 무게는 이보다 훨씬 무겁다 이유를 살펴보기 위해 산점도를 그려본다
import matplotlib.pyplot as plt
distances, indexes = knr.kneighbors([[50]]) #길이 50의 이웃을 구함
plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker = 'D')
plt.scatter(50, 1033, marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
#그래프에 보이듯이 주위  45cm 농어들의 무게평균들로 50cm 농어들의 길이를 구하면 1033이 나온다. 똑같이 100cm 농어의 무게를 구하더라도 똑같은 결과가 나오게된다.
#이러한 문제를 해결하기 위해 선형 회귀 모델을 이용한다.

from sklearn.linear_model import LinearRegression
lr = LinearRegression() #선형 회귀 모델 객체를 만듬
lr.fit(train_input, train_target)
print(lr.predict([[50]])) #[1241.83860323]가 나옴
print(lr.coef_, lr.intercept_)  #lr.coef_가 y=ax+b의 a(기울기), lr.intercept_가 b(y절편)이다. 이를 머신러닝 알고리즘이 찾은 값이라는 의미로 모델 파라미터라고 부른다.
#농어의 길이 15에서 50까지 직선으로 그려보기 위해서 앞에서 구한 기울기와 절편을 사용하여 (15, 15*39-709)와 (50,50*39-709) 두 점을 이으면 된다.
plt.scatter(train_input, train_target)
plt.plot([15, 50], [15*lr.coef_+lr.intercept_,50*lr.coef_+lr.intercept_]) #15에서 50까지 1차 방정식 그래프를 그림
plt.scatter(50, 1241.8, marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show() #그래프에 보이는 직선이 선형 회귀 알고리즘이 이 데이터셋에서 찾은 최적의 직선. 길이가 50cm인 농어의 무게는 이 그래프의 끝에 있다.
#이제 이 선형 회귀 알고리즘이 쓸만한지 훈련 세트와 테스트 세트에 대해 결정계수 R^2를 구한다.
print(lr.score(train_input, train_target)) #훈련 세트
print(lr.score(test_input, test_target)) #테스트 세트
#결과를 보면 전체적으로 수치가 낮은 과소적합 되었다. 그리고 선형 그래프 아래쪽을 보게 되면 이상하다는 것을 알 수 있다.
#사실 농어의 길이와 무게 그래프는 일직선보다는 곡선에 가깝다. 2차 그래프를 그리기 위해선 길이를 제곱한 항이 훈련 세트에 추가되어야 한다.
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input)) #column_stack()함수를 이용하여 두 배열을 나란히 붙인다.
#print(train_poly.shape, test_poly.shape) 길이를 제곱하여 왼쪽 열에 추가했기 때문에 훈련 세트와 테스트 세트 모두 열이 2개로 늘어났다. (42, 2) (14, 2)
#train_poly를 이용해 선형 회귀 모델을 다시 훈련한다. 타깃값은 어떤 그래프를 훈련하든 바꿀 필요가 없다.
lr = LinearRegression()
lr.fit(train_poly, train_target)
#print(lr.predict([[50**2, 50]])) [1573.98423528]가 나오게 되며 1차방정식 값보다 높게 나온다
#print(lr.coef_, lr.intercept_) #기울기와 절편
#이런 방정식을 다항식이라 부러며 다항식을 사용한 선형 회귀를 다항 회귀라고 부른다.
point = np.arange(15, 50) #구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만든다.
plt.scatter(train_input, train_target)
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05) #15에서 49까지 2차방정식 그래프
plt.scatter(50, 1574, marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show() #일직선 그래프보다 훨씬 나은 그래프가 나오게 된다. 그 다음 훈련 세트와 테스트 세트의 R^2 점수를 평가함
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target)) # 테스트 세트가 훈련 세트보다 높으므로 과소적합이다. 따라서 더 복잡한 모델이 필요하다

