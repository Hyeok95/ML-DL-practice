# ML-practice
🚩 머신러닝 공부 🚩

## 1) 지도 학습 알고리즘
* 정의 : 정답이 있는 데이터를 활용해 데이터를 학습시키는 것
* 회귀와 분류로 나뉨
   * 회귀 : 연속하는 숫자를 예측하는 분야
   * 분류
      * 이진 분류 : 2가지 중 하나로 분류하는 것(활성화 함수 : 시그모이드 함수, 손실 함수 : 로지스틱 손실 함수)
      * 다중 분류 : 3가지 이상 중 하나로 분류하는 것(활성화 함수 : 소프트맥스 함수, 손실 함수 : 크로스 엔트로피 손실 함수)

### 1. 선형 회귀
* 데이터 모델에 가장 적합한 선을 찾기위해 데이터들의 점을 사용.
* 링크

https://github.com/Hyeok95/ML-DL-practice/blob/main/ML/2.%20Linear_Regression.ipynb

### 2. 로지스틱 회귀
* 시그모이드 함수일 때 사용하며, 발생할 사건의 확률을 0부터 1까지의 범위로 표현할 때 값을 변환하기 위한 함수로 사용.
* 링크

https://github.com/Hyeok95/ML-DL-practice/blob/main/ML/3.%20Rogistic_Regression.ipynb


### 3. SVM
* 각 데이터의 점들을 선을 사용해서 구분지으며, 이 선은 데이터의 범주를 구분 지을 수 있는 가장 가까운 2개의 데이터의 점을 기준으로 만들어짐.
* 링크

https://github.com/Hyeok95/ML-DL-practice/blob/main/ML/SVM.ipynb


### 4. Naive Bayes
* 데이터가 각 클래스에 속할 특징 확률을 계산하는 조건부 확률 기반의 분류 방법.
* 링크

https://github.com/Hyeok95/ML-DL-practice/blob/main/ML/Naive_Bayes.ipynb

### 5. K-Nearest Neighbor
* 어떤 데이터가 주어지면 그 주변의 데이터를 살펴본 뒤 더 많은 데이터가 포함되어 있는 범주로 분류하는 방식
* 링크

https://github.com/Hyeok95/ML-DL-practice/blob/main/ML/K-Nearest-Neighbor.ipynb

### 6. 의사결정나무
* 변수들로 기준을 만들고 이것을 통하여 샘플을 분류하고 분류된 집단의 성질을 통하여 추정하는 모형. (트리구조)
* 장점 : 해석력이 높고, 직관적, 범용성
* 단점 : 높은 변동성, 샘플에 민감함.
