bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
#도미 데이터11
기
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9];
#빙어 데이터1

import matplotlib.pyplot as plt #matplotlib의 pylot함수를 plt로 줄여서 사용

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length') #x축
plt.ylabel('weight') #y축
plt.show()

length = bream_length+smelt_length  #도미길이+빙어길이
weight = bream_weight+smelt_weight  #도미무게+빙어무게

fish_data = [[l, w] for l, w in zip(length, weight)] # zip함수를 이용해 2차원리스트로 만듬(사이킷런패키지를 사용하기위해 2차원리스트로 만들어야함)


fish_target = [1]*35 + [0]*14 #찾으려는 대상을 1로 두고 아닌것을 0으로 둠


from sklearn.neighbors import KNeighborsClassifier #사이킷런 패키지에서 k-최근접 이웃 알고리즘을 구현한 클래스
kn = KNeighborsClassifier() #임포트한 클래스의 객체 생성
kn.fit(fish_data, fish_target) #학습 훈련  fit()
kn.score(fish_data, fish_target) #얼마나 잘 훈련되었는가 0~1까지 숫자 1이면 완벽 0이면 하나도 못맞춤

kn.predict([[30, 600]])

kn49 = KNeighborsClassifier(n_neighbors=49) #가까운 49개의 데이터를 사용
kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target)

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)

for n in range(5, 50):
    # 최근접 이웃 개수 설정
    kn.n_neighbors = n
    # 점수 계산
    score = kn.score(fish_data, fish_target)
    # 100% 정확도에 미치지 못하는 이웃  출력
    if score < 1:
        print(n, score)
