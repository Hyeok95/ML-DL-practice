# ML-practice
🚩 머신러닝 공부 🚩

## 1) 지도 학습 알고리즘
* 정의 : 정답이 있는 데이터를 활용해 데이터를 학습시키는 것
* 회귀와 분류로 나뉨
   * 회귀 : 연속하는 숫자를 예측하는 분야
   * 분류
      * 이진 분류 : 2가지 중 하나로 분류하는 것(활성화 함수 : 시그모이드 함수, 손실 함수 : 로지스틱 손실 함수)
      * 다중 분류 : 3가지 이상 중 하나로 분류하는 것(활성화 함수 : 소프트맥스 함수, 손실 함수 : 크로스 엔트로피 손실 함수)

### 1. [선형 회귀](https://github.com/Hyeok95/ML-DL-practice/blob/main/ML/2.%20Linear_Regression.ipynb)
* 데이터 모델에 가장 적합한 선을 찾기위해 데이터들의 점을 사용.

### 2. [로지스틱 회귀](https://github.com/Hyeok95/ML-DL-practice/blob/main/ML/3.%20Rogistic_Regression.ipynb)
* 시그모이드 함수일 때 사용하며, 발생할 사건의 확률을 0부터 1까지의 범위로 표현할 때 값을 변환하기 위한 함수로 사용.


### 3. [SVM](https://github.com/Hyeok95/ML-DL-practice/blob/main/ML/SVM.ipynb)
* 각 데이터의 점들을 선을 사용해서 구분지으며, 이 선은 데이터의 범주를 구분 지을 수 있는 가장 가까운 2개의 데이터의 점을 기준으로 만들어짐.


### 4. [Naive Bayes](https://github.com/Hyeok95/ML-DL-practice/blob/main/ML/Naive_Bayes.ipynb)
* 데이터가 각 클래스에 속할 특징 확률을 계산하는 조건부 확률 기반의 분류 방법.

### 5. [K-Nearest Neighbor](https://github.com/Hyeok95/ML-DL-practice/blob/main/ML/K-Nearest-Neighbor.ipynb)
* 어떤 데이터가 주어지면 그 주변의 데이터를 살펴본 뒤 더 많은 데이터가 포함되어 있는 범주로 분류하는 방식

### 6. [의사결정나무](https://github.com/Hyeok95/ML-DL-practice/blob/main/ML/Decision_Tree.ipynb)
* 변수들로 기준을 만들고 이것을 통하여 샘플을 분류하고 분류된 집단의 성질을 통하여 추정하는 모형. (트리구조)
* 장점 : 해석력이 높고, 직관적, 범용성
* 단점 : 높은 변동성, 샘플에 민감함.

### 🚩앙상블(Ensenble)
* 여러 개의 기본모델을 활용하여 하나의 새로운 모델을 만들어내는 개념이며 학습방법이 가장 불안정한 의사결정나무에 주로 이용한다.
* 종류 : Bagging, RandomForest, Boosting(Adaboost, Gradient boosting), Stacking 등이 있다.

#### ☀ [Bagging](https://github.com/Hyeok95/ML-DL-practice/blob/main/ML/Ensenble%2001.-%20Bagging.ipynb)
* 주어진 자료에서 여러 개의 Bootstrap 자료를 생성하고 각 붓트스랩 자료에 예측모형을 만든 후 결합하여 최종 예측모형을 만드는 방법이다. Voting은 여러 개의 모형으로부터 산출된 결과 중 다수결에 의해서 최종 결과를 선정하는 과정이다.

#### ☀ [RandomForest](https://github.com/Hyeok95/ML-DL-practice/blob/main/ML/Ensenble%2002%20-%20Randomforest.ipynb)
* 여러개의 모델을 다양하게 만들기 위해 데이터와 변수도 재구성하는 방법 (Bagging보다는 성능이 뛰어남)

#### ☀ Boosting
* 예측력이 약한 모형들을 결합하여 강한 예측모형을 만드는 방법

> [Adaboost](https://github.com/Hyeok95/ML-DL-practice/blob/main/ML/Ensemble%2003%20-%20Adaboost.ipynb)
* 기본 모델이며 오분류된 데이터에 가중치를 주어 완벽하게 분류하는 모델.

> [Gradient boosting](https://github.com/Hyeok95/ML-DL-practice/blob/main/ML/Ensemble%2004%20-Gradient_Boosting.ipynb)
* Adaboost에서 분류한 모델이 예측한 데이터의 오차를 가지고 이 오차를 예측하는 weak learner를 만들고 weak learner들을 결합하여 성능이 좋은 모델을 만드는 것.

   * XGBoosting : Regularization term이 추가되어 모델이 과적합 되는것을 방지
   * LightGBM : Xgboost와 다르게 leaf-wise를 loss사용하여 tree의 뿌리를 조금 더 깊게 내릴수 있음.(loss를 더 줄일 수 있음)
   * Catboost : 분산과 편향을 줄이는 기법이며, categorical feature를 수치형으로 변환하는 방법을 사용함. (오픈소스가 다소 부족)


## 2) 비지도 학습 알고리즘

* 정의 : 지도 학습과는 달리 정답 라벨이 없는 데이터를 비슷한 특징끼리 군집화 하여 새로운 데이터에 대한 결과를 예측하는 방법
* 종류 : Clustering,  K Means,  Density Estimation,  Exception Maximization,  Pazen Window  DBSCAN

### 1. 군집분석(Clustering)
* 각 데이터의 유사성을 측정하여 높은 대상 집단을 분류하고, 군집 간에 상이성을 규명하는 방법
  * [K-means clustering](https://github.com/Hyeok95/ML-DL-practice/blob/main/ML/Kmeans_clustering.ipynb) : 데이터의 사용자가 지정한 k개의 군집으로 나눔
    1. 각 데이터 포인트 i에 대해 가장 가까운 중심점을 찾고, 그 중심점에 해당하는 군집 할당
    2. 할당된 군집을 기반으로 새로운 중심 계산, 중심점은 군집 내부 점들 좌표의 평균으로 함.
    3. 각 클러스터의 할당이 바뀌지 않을 때까지 반복
    4. K-medoids clustering(k-means clusering의 단점을 보완)
      * 군집의 무게중심을 구하기 위해 데이터의 평균 대신 중간점을 사용(K-meas보다 이상치에 강건한 성능을 보임)
  * Hierarchical clustering(계층적 군집분석) : 나무 모양의 계층 구조를 형성해 나가는 방법
  * DBSCAN : k개를 설정할 필요없이 군집화 할 수 있는 방법
