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