fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

import numpy as np #데이터가 많은 경우엔 리스트로 작업하는 것보다 넘파이 배열을 이용하는 것이 효율적이다.
#np.column_stack(([1,2,3],[4,5,6])) column_stack() 함수는 전달받은 리스트를 일렬로 세운 다음 차례대로 나란히 연결함(튜플로 전달)
#array([[1,4],[2,5],[3,6]])
fish_data = np.column_stack((fish_length, fish_weight))
#print(fish_data[:5]) 잘만들어졌는지 확인
fish_target = np.concatenate((np.ones(35), np.zeros(14)))  #np.concatenate() 첫번째 차원을 따라 배열을 연결 np.ones(), np.zeros()는 각각 1과 0의 배열을 만듬
#print(fish_target) 만들어진 배열 확인 

from sklearn.model_selection import train_test_split
#train_test_split() 이 함수는 전달되는 리스트나 배열을 비율에 맞게 훈련 세트와 테스트 세트로 섞어서 나누어줌 (model_selection 모듈 아래 있음)
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42) #random_state는 랜덤시드와 같은 역할 (테스트 세트로 전체의 약 25%를 떼어냄)
#print(train_input.shape, test_input.shape) 데이터의 크기 확인 
#print(train_target.shape, test_target.shape) 위쪽과 아래쪽 둘 다 튜플의 형태로 나옴
#print(test_target)
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify= fish_target, random_state=42) #훈련 데이터가 작거나 특정 클래스의 샘플 개수가 적을 때 stratify 매개변수에 타깃데이터 전달
#print(test_target) 비율 맞춘 후 확인

#위에서 준비한 데이터로 k-최근접 이웃을 훈련해봄
from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)
#print(kn.predict([[25,150]])) 길이 25, 무게 150인 도미를 예측해봄 그러나 0이 나옴 그래서 산점도를 통해 위치를 확인해봄

import matplotlib.pyplot as plt #그래프 그리기
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

distances, indexes = kn.kneighbors([[25, 150]]) #25, 150 근처(기본 5개)를 찾아주는 kneighbors() 메서드
 
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D') #marker='D'로 하게 되면 마름모로 나타남
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

#print(train_input[indexes])
#print(train_target[indexes])
#print(distances)

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
plt.xlim(0,1000) #xlim()함수는 x축 범위설정
plt.xlabel('length')
plt.ylabel('weight')
plt.show() #그래프에서 볼 수 있듯이 데이터들을 예측하기 위해선 데이터의 특성값을 일정한 기준으로 맞춰 주어야 함 이런 작업을 데이터 전처리라고 함(특히 k-최근접 이웃처럼 거리 기반으로 예측하는 경우)
#가장 널이 사용하는 전처리 방법 중 하나는 표준점수(z점수라고도 함)

mean = np.mean(train_input, axis=0) #평균을 계산
std= np.std(train_input, axis=0) #표준편차를 계산 특성마다 스케일이 다르므로 각각 계산하기 위해 axis=0으로 함
#print(mean, std)
train_scaled = (train_input - mean) / std  #표준점수= (데이터-평균) / 표준편차

new = ([25, 150] - mean) / std
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

kn.fit(train_scaled, train_target)
test_scaled = (test_input-mean) / std
kn.score(test_scaled, test_target)
#print(kn.predict([new])) 도미로 예측함

distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1],marker='^')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
