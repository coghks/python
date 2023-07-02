import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()
#print(pd.unique(fish['Species'])) Species열에서 고유한 값을 추출하기 위한 unique()함수
#이 데이터프레임에서 Species열을 타깃으로 만들고 나머지 5개 열은 입력 데이터로 사용함
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy() #넘파이 배열로 바꿔서 fish_input에 저장
#print(fish_input[:5]) 처음부터 5개만 출력
fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
from sklearn.preprocessing import StandardScaler
ss= StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input) #훈련 세트와 테스트 세트를 표준화 전처리

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors= 3)
kn.fit(train_scaled, train_target)
#print(kn.score(train_scaled, train_target))
#print(kn.score(test_scaled, test_target)) #훈련 세트와 테스트 세트에 대한 점수
#print(kn.classes_) ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish'] #타깃값 순서
#print(kn.predict(test_scaled[:5])) 테스트 세트에 있는 처음 5개 샘플의 타깃값 예측 4번째 샘플 확률[0.     0.     0.6667 0.     0.3333 0.     0.    ]

import numpy as np
proba = kn.predict_proba(test_scaled[:5]) #predict_proba() 메서드로 클래스별 확률값을 반환
#print(np.round(proba, decimals=4)) #기본적으로 round() 함수는 소수점 첫째 자리에서 반올림을 하는데, dicimals 매개변수로 유지할 소수점 아래 자릿수를 지정할 수 있다.
#kneighbors() 메서드의 입력은 2차원 배열이어야 한다. 이를 위해 넘파이 배열의 슬라이싱 연산자를 사용 슬라이싱 연산자는 하나의 샘플만 선택해도 항상 2차원 배열이 만들어진다.
#여기서는 4번째 샘플의 최근점 이웃의 클래스를 확인
distances, indexes = kn.kneighbors(test_scaled[3:4])
#print(train_target[indexes]) [['Roach' 'Perch' 'Perch']] 위와 같은 결과가 나온다.

#로지스틱 회귀
import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5, 5, 0.1)
phi = 1/ (1+np.exp(-z)) #확률이므로 값을 0에서 1로 제한하기 위해 시그모이드 함수를 사용
plt.plot(z,phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

#로지스틱 회귀로 이진 분류 수행하기
#이진 분류일 경우 시그모이드 함수의 출력이 0.5보다 크면 양성 클래서, 0.5보다 작으면 음성 클래스로 판단(사이킷런은 0.5일경우 음성 클래스로 판단)
#넘파이 배열은 True, False 값을 전달하여 행을 선택할 수 있다. 이를 블리언 인덱싱이라고 함
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt') #도미와 빙어에 대한 행만 골라냄
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
#print(lr.predict(train_bream_smelt[:5])) ['Bream' 'Smelt' 'Bream' 'Bream' 'Bream'] 두 번째 샘플을 제외하고는 모두 도미로 예측
#예측확률은 predict_proba()메서드에서 제공
#print(lr.predict_proba(train_bream_smelt[:5])) #train_bream_smelt에서 처음 5개 샘플의 예측 확률    샘플마다 2개의 확률이 출력, 첫 번째 열이 음성 클래스(0)에 대한 확률이고 두 번째 열이 양성 클래스(1)
#print(lr.classes_) ['Bream' 'Smelt'] 빙어(Smelt)가 양성 클래스
#print(lr.coef_, lr.intercept_) [[-0.4037798  -0.57620209 -0.66280298 -1.01290277 -0.73168947]] [-2.16155132] 학습한 계수
#이 로지스틱 회귀 모델이 학습한 방정식은 다음과 같다
# z = -0.404*(Weight) -0.576*(Length) -0.663*(Diagonal) -1.013*(Height) -0.732*(Width) -2.161
decisions = lr.decision_function(train_bream_smelt[:5])
#print(decisions) [-6.02927744  3.57123907 -5.26568906 -4.24321775 -6.0607117 ] 이 z값들을 시그모이드 함수에 통과시키면 확률을 얻을 수 있다.
from scipy.special import expit #사이파이 라이브러리에 있는 시그모이드 함수 expit()
#print(expit(decisions)) [0.00240145 0.97264817 0.00513928 0.01415798 0.00232731] 
#출력된 값을 보면 predict_proba() 메서드 출력의 두 번째 열의 값과 동일하다. 즉 decision_function() 메서드는 양성 클래스에 대한 z값을 반환

#로지스틱 회귀로 다중 분류 수행하기
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
#print(lr.score(train_scaled, train_target)) 훈련 세트에 대한 점수 0.9327731092436975
#print(lr.score(test_scaled, test_target)) 테스트 세트에 대한 점수 0.925
#print(lr.predict(test_scaled[:5])) 테스트 세트의 처음 5개 샘플 ['Perch' 'Smelt' 'Pike' 'Roach' 'Perch']
proba = lr.predict_proba(test_scaled[:5])
#print(np.round(proba, decimals=3)) 테스트 세트 5개 샘플에 대한 예측 확률
#print(lr.coef_.shape, lr.intercept_.shape) (7, 5) (7,) 이 데이터는 5개의 특성을 사용하므로 coef_배열의 열은 5개 그런데 행이 7개 intercept_도 7개 즉 z값을 7개나 계산함
#다중 분류는 클래스마다 z값을 하나씩 꼐산 가장 높은 z값을 출력하는 클래스가 예측 클래스가 된다.
#확률은 이진 분류에서는 시그모이드 함수를 사용해 z를 0과 1 사이의 값으로 변환 했지만 다중 분류는 이와 달리 소프트맥스 함수를 사용하여 7개의 z값을 확률로 변환(정규화된 지수 함수라고도 함)
decision = lr.decision_function(test_scaled[:5])
#print(np.round(decision, decimals=2)) #테스트 세트의 처음 5개 샘플에 대한 z1~z7의 값
from scipy.special import softmax
proba = softmax(decision, axis=1)  #앞서 구한 decision배열을 softmax() 함수에 전달 후 softmax()의 axis 매개변수는 소프트맥스를 꼐산할 축을 지정 여기에서는 axis=1로 지정하여 각 행, 즉 각 샘플에 대해 소프트맥스를 계산
#만약 axis 매개변수를 지정하지 않으면 배열 전체에 대해 소프트맥스를 계산함
#print(np.round(proba, decimals=3)) 출력 결과를 앞서 구한 proba 배열과 비교해보면 결과가 정확히 일치한다.
어려웡

