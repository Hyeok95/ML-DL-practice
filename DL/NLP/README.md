# 자연어 처리(NLP) 

## 1. NLP WorkFlow
* NLP : 컴퓨터를 이용해 사람의 언어를 분석하고 처리하는 기술.
(자연어 이해(NLU) + 자연어 생성(NLG))
* NLU : 자연어를 컴퓨터가 이해할 수 있는 값으로 바꾸는 과정
* NLG : 컴퓨터가 이해한 값을 사람이 이해할 수 있도록 바꾸는 과정

데이터 수집 -> 데이터 정제 -> 데이터 라벨링 -> Tokenization -> Segmentation -> Batchify

> 데이터 수집
    
* 자연어 처리를 위한 데이터는 오픈된 데이터가 많이 없어서 크롤링을 이용하여 주로 수집한다. 하지만 쓰레기 데이터가 많이 섞여 있어서 정제 과정이 매우 힘드므로 신경을 써야함.

> 데이터 정제
* 정규 표현식을 주로 사용함.
    
    정규식이 설정 되어 있을 때 하나라도 들어가 있는 문장은 매치에 성공한다.
    ([], ^..$, -, ^, ?, +, *, {})
* 대소문자를 통일한다.

> 데이터 라벨링
* Text Classification
    * 입력 : 문장 / 출력 : 클래스
* Token Classification
    * 입력 문장 / 출력 : 형태소 등
* Sequence to Sequence
    * 입력 문장 / 출력 : 문장

> Tokenizaion
* 토큰화는 Sentence Segmentation과 tokenization으로 구성.
* Sentence Segmentation : 우리가 수집한 문장에 대해 원하는 형태로 변환하는 것
* Tokenization은 두 개 이상의 다른 토큰들로 이루어진 단어를 쪼개어 단어의 개수를 줄이고, 희소설을 낮추기 위함.

(영어 : 형태소 분석, 한글 : Mecab, KoNLPy를 주로 사용)

## 2. 단어의 표현 방법
> 원-핫 인코딩
* 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식.<br>
[한계]<br>
단어의 개수가 늘어날 수록 벡터를 저장하기 위해 필요한 공간이 계속 늘어나는 단점을 가짐.

> Word Embedding(워드 임베딩)
* 단어를 벡터로 표현하여 밀집 표현으로 변환하는 방법.

> Word2Vec(워드투벡터)
* 주변 단어로 해당 단어를 학습시키는 알고리즘
(분산 표현 : 주변에 있으면 단어의 의미가 더 비슷할거라 가정)
    * CBOW : 주변에 있는 단어들을 가지고, 중간에 있는 단어들을 예측하는 방법
    * Skip-Gram : 중간에 있는 단어로 주변 단어들을 예측하는 방법


## 3. RNN(순환신경망)

> RNN<br>
* 입력과 출력을 시퀀스 단위로 처리하는 시퀀스 모델.

<img src="https://github.com/Hyeok95/ML-DL-practice/blob/main/DL/NLP/Image/RNN.PNG">

* Sequence model

<img src="https://github.com/Hyeok95/ML-DL-practice/blob/main/DL/NLP/Image/RNN2.PNG">

1. one to many<br>
이미지 데이터에서 설명글을 출력함 ex) image captioning
2. many to one<br> 
텍스트에서 감정이나 부정적 분류, 최근 날씨에 따른 향후 날씨 예측 ex) sentiment, 시계열 예측<br>
3. many to many<br>
영어문장을 한글로 번역, 영상 프레임별로 예측 ex)sequnce of words, vidio calssification

* RNN의 과정
    - input<br>
우선 시퀀스 데이터 즉 정보는 벡터로 표현되어 model에 input됩니다.
(가령 TEXT인 경우는 벡터로 표현하기 위해 Word embedding이라는 작업을 합니다.)
    - state<br>
input인 벡터(가공된 정보)는 순서(Timestep)에 따라 순환신경망의 상태(State, 초록색 상자)에 저장됩니다. 이 상태(state)는 함수로 변형될 수 있습니다. 즉 함수(f)는 파라미터 W(가중치)와 이전 상태(old state), input로 구성되어 있습니다.
    - update<br>
 W가 튜닝되면서 계속 발전(Update)해 나갑니다.
    - output<br>
매 Timestep마다 새로운 인풋이 들어오면서 다른 반응 결과를 출력함.

* RNN 특이사항
    - 매 time step마다 동일한 funtion과 동일한 parameters 사용되어야함.

h = W(xt) + b +Uh(t-1)
* h : hiden layer, x : 입력, y : 출력

X(t) : 현재 입력 벡터 값<br>
W(xh) : 입력(x)에서 hiden layer사이의 파라미터<br>
h(t-1) : 이전 timestep hiden layer의 상태<br>
W(hh) : 현재 hiden layer와 직전의hiden layer 사이에서의 파라미터<br>
h(t) : 현재 timestep hiden layer의 상태<br>
W(hy) : hiden layer에서 출력(y) 사이의 파라미터<br>


> LSTM<br>
* 오래 전 혹은 최근의 단어들을 기억할 수 있는 아키텍쳐가 LSTM이다.

<img src="https://github.com/Hyeok95/ML-DL-practice/blob/main/DL/NLP/Image/LSTM.PNG">

- LSTM의 과정
    - state : RNN과 같습니다. 무엇이 다르냐? cell state가 추가됩니다.<br>
    즉 LSTM의 state는 hiden state와 cell state 총 2개입니다. input인 벡터(가공된 정보)가 순서(Timestep)에 따라 이전 상태(old hiden state)와 함께 현재 상태(hiden State)에 저장되기 앞서 기억상태(cell state)가 참전하는 처리과정을 겪습니다.
- 이 처리과정에서는 3개의 gate(i, f, o)가 cell state의 참전(활성화)여부를 결정합니다.

> GRU<br>

<img src="https://github.com/Hyeok95/ML-DL-practice/blob/main/DL/NLP/Image/GRU.PNG">

* 기존 LSTM구조를 조금 더 간단하게 개선한 모델입니다.
    - LSTM과 다르게 reset gate, update gate 2개의 gate만을 사용합니다
    - Reset gate : 이전 시점의 hidden state와 현 시점의 x를 활성화함수 Sigmoid를 적용하여 구하는 방식
    - update gate : 과거와 현재의 정보를 가각 얼마나 반영할지에 대한 비율을 구함.


## 4. Transformer
> 시퀀스-투-시퀀스(Sequence-to-Sequence)

* 입력된 시퀀스로부터 다른 도메인의 시퀀스를 출력하는 다양한 분야에서 사용되는 모델
- 단점
1. 하나의 고정된 크기의 벡터에 모든 정보를 압축하려고 하니까 정보 손실이 발생합니다.
2. RNN의 고질적인 문제인 기울기 소실(vanishing gradient) 문제가 존재합니다.

> Attention<br>
* 디코더에서 출력 단어를 예측하는 매 시점(time step)마다, 인코더에서의 전체 입력 문장을 다시 한 번 참고한다는 점
Attention(Q, K, V) = Attention Value<br>
Q = Query : t 시점의 디코더 셀에서의 은닉 상태<br>
K = Keys : 모든 시점의 인코더 셀의 은닉 상태들<br>
V = Values : 모든 시점의 인코더 셀의 은닉 상태들<br>
쿼리(Query)'에 대해서 모든 '키(Key)'와의 유사도를 각각 구합니다. 그리고 구해낸 이 유사도를 키와 맵핑되어있는 각각의 '값(Value)'에 반영해줍니다.

* **Scaled dot-product Attention(스케일드 닷-프로덕트 어텐션)**<br>
함수를 사용하는 어텐션을 어텐션 챕터에서 배운 닷-프로덕트 어텐션(dot-product attention)에서 값을 스케일링하는 것을 추가

- **Multi-head Attention** <br>
h개의 각각 다르게 초기화된 parameter matrix를 곱하여 h개의 어텐션 결과를 얻고 이를 concat한 다음 다른 weight matrix를 곱하여 최종적인 어텐션 값을 출력

- **Positional Encoding** <br>
트랜스포머는 단어를 순차적으로 받는 것이 아니라 병렬적으로 처리합니다. 하지만 layer를 통과하면서 위치에 대한 정보가 없다면 단어의 위치에 대한 고려를 할 수 없을 것이고, 이를 위해 첫 인코딩 레이어를 통과하기 전 positional encoding을 해줍니다.


> Transformer<br>
* 기존의 seq2seq의 구조인 인코더-디코더를 따르면서도, 논문의 이름처럼 어텐션(Attention)만으로 구현한 모델

> BERT(Bidirectional Encoder Representations from Transformers)
 * 문장 중간에 빈칸을 만들고 해당 빈칸에 어떤 단어가 적절할지 맞추는 과정에서 프리트레인합니다. (양방향)

- **마스크드 언어 모델(Masked Language Model, MLM)** <br>
80%의 단어들은 [MASK]로 변경한다. <br>
Ex) The man went to the store → The man went to the [MASK] <br>
10%의 단어들은 랜덤으로 단어가 변경된다. <br>
Ex) The man went to the store → The man went to the dog <br>
10%의 단어들은 동일하게 둔다. <br>
Ex) The man went to the store → The man went to the store <br>

- **다음 문장 예측(Next Sentence Prediction, NSP)**

> GPT <br>
* 이전 단어들이 주어졌을 때 다음 단어가 무엇인지 맞추는 과정에서 프리트레인(pretrain)합니다 (단방향)

* **차이** <br>
1. GPT는 문장 생성에, BERT는 문장의 의미를 추출하는 데 강점을 지닌 것으로 알려져 있습니다.
2. BERT는 트랜스포머에서 인코더(encoder), GPT는 트랜스포머에서 디코더(decoder)만 취해 사용한다는 점

