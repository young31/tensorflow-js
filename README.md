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
