# Tensorflow JS
- 엣지 디바이스에서 머신러닝을 위한 툴
- 2020_Book_PracticalTensorFlowJs.pdf 책을 보고 실습 및 공부한 내용을 정리하는 저장소입니다.
- html 및 javascript에 대한 지식이 매우 낮지만 필요한 부분만 이해하면서 진행할 예정입니다.
- 최종목표는 웹상에서 모형 구축 및 적용(inference)
    - 서버에서 구축된 모형과의 interaction
    - 엣지 디바이스에서의 개별 최적화

## 1. Modeling
- 간단한 logistic, linear regression 모형을 구현해보는 실습
- sequential을 사용하는 api로 모형 구현
- 미리 정의된 데이터 셋을 불러와서 훈련해보는 수준의 실습
    - 추후에는 dataloader를 만들어서 직접 데이터를 업로드하게 하면 좋을 것 같다.

## 2. Clustering
- ml5를 이용하여 클러스터링 하는 예제이다.
- ml5를 찾아보니 js상에서 활용할 수 있는 high-level 라이브러리인것 같다.
- 유용할 때가 있을 것 같으나 너무 high-level이라 직접쓰기에는 적절치 않을 것 같다.
    - 일종의 서브 툴팁 정도로 활용하면 좋을 것 것 같다.

## 3. CNN (mnist)
- local에서 데이터를 불러와서 훈련시키는 내용을 배운다.
- 사용하던 pandas나 numpy를 안쓰니 데이터 받는 부분도 생각보다 어렵다.
- 모델을 정의하는 방식은 기존하고 비슷해서 응용하기는 어렵지 않은 것 같다.
    - 대소문자가 달라서 헷갈리는 부분들만 주의하자!
- tf.tidy(() => {function}) , {tensor}.dataSync() 등이 처음 보는 개념이니 숙지해보도록 하자.

## 4~5. Pretrained Model
### 4. PoseNet
- 모델을 개발하기보다는 미리 구현되어 있는 부분을 활용하는 부분이다.
- 모델 개발에 관심이 있는 입장에서는 그리 중요한 정보가 아니나, 개발 단계에서는 유용할 것으로 보임
- 어디서 왔는지 살펴보니 [여기](https://www.tensorflow.org/js/models?hl=ko) 에서 다양한 목적에 맞는 모형을 제공하는 것을 확인할 수 있다.
- 추후에는 해당 코드를 보면서 모델링 방법 및 배포하여 활용하는 법을 볼 수 있을 것 같다.
- 관련 기술: js에서 캠을 이용하여 사진 스트리밍(동영상)  
~~js도 어렵고 html도 어려운데 visualize는 너무 어렵다.~~
### 5. Toxicity
- 구현되어 있는 모델을 사용한다.
- 대부분의 값들이 미리 정의된 값들로 부터 사용된다.
    - 모델을 구성하여 미리 잘 정의하면 활용도가 높을 수 있음을 느꼈다.
- 생각보다 extension 만들기가 어렵지 않다.
    - 추후에 어플리케이션을 만들어보면 좋을 것 같다.
    ~~깊이 들어가면 어렵겠지?~~
- 관련 기술: google extension

## 6. Load locally trained model from google cloud
- 모형을 완전히 구현해서 가져온 것은 아니고, google cloud에서 학습시킨 포맷으로만 가능한 코드이다.
- google cloud를 통해서 export하는 모델은 특별한 규칙으로 formatting 되어 있어서 파싱하는 코드가 따로 존재하는 듯 하다.
- weight를 bin형식으로 저장해서 받아올 수 있다.
- 관련 기술: js에서 캠을 이용해서 동영상, google auto-ml로 훈련한 모델 불러오기  
~~곧 원하는 내용이 나올 것 같으니 조금만 힘내자~~
