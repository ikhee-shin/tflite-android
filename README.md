# ONE 을 이용한 ANDROID 환경에서 TEXT SENTIMENT CLASSIFICATION

목표

- ONE-nnpackage 로 돌아가는 TEXT SENTIMENT CLASSIFICATION 안드로이드 앱 핸드폰에서 구동

과정

- [tensorflow examples](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android)  을 따라서 tflite 로 구동되는 안드로이드 앱 핸드폰에서 우선 구동
- 그 후 tflite 모델을 nnpackage 로 바꾸고 핸드폰에서 구동



## ONE 준비

https://github.com/Samsung/ONE

[ONE 1.6.0 RELEASE](https://github.com/Samsung/ONE/releases/tag/1.6.0) 에서 아래 파일들 받기

- [nnfw-1.6.0-android-aarch64.tar.gz](https://github.com/Samsung/ONE/releases/download/1.6.0/nnfw-1.6.0-android-aarch64.tar.gz)
  - 헤더, 라이브러리
  - 3rdparty/android 에 준비
- [nncc-1.6.0.tar.gz](https://github.com/Samsung/ONE/releases/download/1.6.0/nncc-1.6.0.tar.gz)
  - bin/model2package 로 tflite 모델 패키징 



## 텐서플로우 모델 준비

https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android



Text Classification 에 쓸 간단한 모델 준비

- https://www.tensorflow.org/tutorials/keras/text_classification_with_hub

- 학습 환경 : tf2.1

- 학습 후 frozen 모델로 저장

  - ```shell
    python3 tensorflow_test_classification/train_model.py
    ```

  - https://github.com/leimao/Frozen_Graph_TensorFlow

- tf2.3.0rc 를 이용해서 tflite 로 변경

  - ```shell
    pip install tensorflow==2.3.0rc0
    python3 tensorflow_test_classification/tf2.3_model_to_tflite.py
    ```

주의할점

- nnpackage 를 안쓴다면 tflite 로 변경할 때 converter.allow_custom_ops = True 해주면 안됨
  - tflite 에 없는 op 이 있을 경우 문제 생길 수 있음
  - layernorm 같은 op 의 경우 tflite 에 구현되어 있지 않은데 nnpackage 를 사용할 경우 이용할 수 있음



## tflite 모델과 안드로이드 앱

핸드폰에 연결해서 하는 방법

- 디버깅 모드 해제해야 했음



안드로이드 스튜디오에서 보는 방법? 

- https://developer.android.com/studio/run/managing-avds?hl=ko
- virtual device 만들면 됨



문제 해결

- probably compressed
  - noCompress 에 추가하기
  - https://github.com/tensorflow/tensorflow/issues/22333