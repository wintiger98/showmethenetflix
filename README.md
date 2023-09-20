# Show me the Netflix

넷플릭스 컨텐츠 흥행 예측 웹애플리케이션

## 소개

이 프로젝트는 Flask를 사용하여 개발한 웹 애플리케이션입니다. Kaggle에서 Netflix movie 데이터 7000여개를 가져와 
영화 이름, 상영등급, 장르, 출시 년도, 출시 날짜, 평점, 투표수, 감독, 작가, 주연배우, 국가, 예산, 수익, 배급사, 상영시간 등의 요인들에 대해 
흥행도 예측 모델을 만들었습니다.
다음과 같은 단계를 거칩니다.
1. Z-score로 정규화
2. 이상치 제거
3. DNN(Deep Neutral Netwrok), Softmax Regression, Random Forest Regression, SVM(Support Vector Regression) 모델 적용
4. 모델별 예측 값 제공

## 화면
1. 홈페이지
![image](https://github.com/wintiger98/showmethenetflix/assets/78953151/a87a5ab6-c03b-4a25-9932-64dbeccd520d)

2. 결과창
![image](https://github.com/wintiger98/showmethenetflix/assets/78953151/f1382bb1-5ec7-4203-bbc6-d1ddf4d582cd)



## 시작하기
```
git clone https://github.com/your/repository.git
cd project-directory
pip install -r requirements.txt
```
실행
```
flask run
```


### 필수 사항
`Python 3.7 이상`
`Flask 라이브러리`
`가상환경 (virtualenv)`
