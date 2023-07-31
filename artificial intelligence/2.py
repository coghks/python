fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0] #길이
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]        #무게

fish_data=[[l,w] for l,w in zip(fish_length,fish_weight)] #2차원리스트로 묶음
fish_target= [1]*35+ [0]*14

from sklearn.neighbors import KNeighborsClassifier #사이킷런 클래스 임포트하고 모델객체생성//
kn=KNeighborsClassifier()

train_input = fish_data[:35] #훈련 세트로 입력값 중 0부터 34번째 인덱스까지 사용
train_target = fish_target[:35] #훈련 세트로 타깃값 중 0부터 34번째 인덱스까지 사용
test_input = fish_data[35:] #테스트 세트로 입력값 중 35번째부터 마지막 인덱스까지 사용
test_target = fish_target[35:] #테스트 세트로 타깃값 중 35번째부터 마지막 인덱스까지 사용

kn = kn.fit(train_input, train_target) #훈련세트로 모델을 훈련시킴
kn.score(test_input, test_target)      #테스트세트로 평가함  그러나 0이나오게 됨(도미와 빙어의 자료가 섞이지않고 샘플링 편향되어있음)

#골고루 섞어주기 위해 numpy를 이용함
import numpy as np

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
#print(input_arr.shape) # 이 명령어를 사용하면 (샘플 수, 특성 수)를 출력함

#순서대로 묶인 input 과 target을 짝맞추어 랜덤으로 섞어주기 위해 arange()함수 사용
np.random.seed(42) #t섞을 때 마다 다른 결과가 나오는데 똑같은 결과를 만들어주기 위해 seed()사용
index = np.arange(49)
np.random.shuffle(index)
#print(index)
#print(input_arr[[1,3]]) #numpy는 슬라이싱 외에 배열 인덱싱이란 기능이 있음(인덱스로 한 번에 여러 개의 원소를 선택가능)

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
#print(input_arr[13], train_input[0])

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

#잘 섞여있는지 산점도를 통해 확인
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1]) #[:.0] ,[:,1] 0과 1의 차이는 2차원리스트 안에 [25.4 ,242.0 ] 0일 경우 앞 1일 경우 뒤를 의미
plt.scatter(test_input[:,0],test_input[:,1])
plt.xlabel("length")
plt.ylabel("weight")
plt.show()
#예비군가기싫다..1일차
kn=kn.fit(train_input, train_target)
kn.score(test_input, test_target)   #1.0이 나옴 100%의 정확도

kn.predict(test_input)
test_target #윗 줄과 동일한 결과가 나옴
