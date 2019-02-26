---
title: "Tensorflow Saving / Tensorflow Serving 정리"   
---

1. Saving
Tensorflow에서 Save를 한다는 것의 의미는 무엇인가?
기본적으로 Tensorflow의 모든 entities는 Protobuf record라는 형식으로 저장/로딩이 된다. Protobuf는 기본적으로는 binary 포맷으로 만들어졌으며, 구글에서 분산처리 시스템에서 통신을 위해 주로 쓰고 있는 파일 형식이다.
이러한 Protobuf record가 디스크에 저장될 때는 Serialized Protobuf record 형식으로 .pb (binary format) 또는 .pbtxt (text format)으로 저장이 된다.
특정 Tensoflow 객체가 Protobuf 형식인지 확인할 수 있는 방법은, 객체 이름에 “Def”가 붙어있는지 확인하는 것이다. 즉, tf.Graph는 실제 모델의 구조를 의미하고, tf.GraphDef는 tf.Graph를 로딩할 수 있는 정보를 담고 있는 Protobuf record (물리적인 파일) 인 것이다. (단, 모든 “Def”가 붙은 객체가 대응되는 Python class를 가지고 있는 것은 아니다. 즉, MetaGraphDef는 모델을 구동하기 위한 Graph 와 부가정보를 담고 있는 Protobuf record를 의미하지만 MetaGraph라는것은 존재하지 않는다.)

Tensorflow에서 Save를 하는 방법은?
1.	Save and Restore variables (변수 저장)
.ckpt와 .meta 파일을 떨어뜨려 저장하는 방식으로, 이는 학습 과정에서 원하는 epoch을 선택하고 싶을 경우 등 모델의 variable의 값 (weight) 대해서 분석이 필요할 경우에 저장하는 방식이다. 즉 model를 저장한다기 보다는 variable을 저장하는 것으로 볼 수 있다.   
2.	Save and Restore models (모델 저장)
SaveModel class를 이용하는 방법으로, variable의 값 (weight) 변화가 관심 있는 것이 아니라, 모델의 학습이 끝난 후, best 모델을 찾아낸 후에 그 모델을 serving하고자 할 때 사용하는 방법이다. SavedModel을 활용하면 다양한 목적의 여러개의 graph를 가진 모델을 저장할 수 있다. 즉, Graph #1은 추론 목적의 그래프이고, Graph #2는 재학습 목적의 그래프로 지정해 놓을 수 있다. Variable 값은 Graph와 별도로 저장이 되는데, 만약 함께 저장된다면 각 Graph에 해당하는 다른 variable이 모두 저장되기 때문이다. 따라서 SavdeModel에서는 variable은 독립적으로 존재하고 여러 Graph에 대응하도록 구성되어있다. 
 
SavedModel로 모델을 저장할 경우에, 하나의 파일로 저장되는 것이 아니라 모델의 정보를 담고 있는 하나의 “폴더”로 저장되는 것으로 이해하는 것이 좋다. 이 폴더에는 assets, variables, saved_model.pb 등의 파일 및 하위폴더 들이 존재하는데, 이중에서 saved_model.pb를 자세히 볼 필요가 있다.
 
-	Assets/ 부가적인 Graph assets을 담고 있는 폴더
-	Assets.extra/ 설명 생략..
-	Vairables/ 모델 학습 결과 (weight) 값을 담고 있는 폴더 (tf.saver.save의 output)
-	Saved.model.pb
SavedMode protobuf로써 실제 MetaGraphDef에서 정의된 graph 정보를 담고 있다.
(참고로, CLI 커맨드를 이용해서 SavedModel의 구조를 알아볼 수 있는 기능도 Tensorflow에서 제공하고 있다.)



그럼 모델을 Saving은 어떻게 하는 것이 효과적인가?
일반적인 실험을 주로 하는 유저들은 weight의 변화에 따른 성능 변화를 보기 위해 tf.train.saver를 이용해서 .ckpt 및 .meta를 떨어뜨리는 방식으로 모델을 생성하고 저장, 로딩 하고 있다. 하지만 최종 모델을 확정 후 배포를 위해서는 Tensorflow에서는 SavedModel을 사용할 것을 적극 권장하고 있다. SavedModel로 배포가 되면 추후 Tensorflow Serving을 통해 유연한 배포모델 관리가 가능하다. 

물론, Freezing graph를 통해 기존 .ckpt 및 .meta에서 불필요한 정보를 제거한 하나의 .pb 파일로 모델을 떨굴 수도 있다. 이러한 방법은 한 개의 파일로 모델의 모든 정보를 담을 수 있다는 장점이 있으나, SavedModel에서 export files라는 기능을 통해 모두 커버하는 기능이고, SavedModel은 signature 및 tag 설정으로 보다 쉬운 모델 배포가 가능하는 측면에서 SavedModel를 쓰는 방법이 권장되고 있다.
SaveModel을 쓸 경우 장점을 정리해보면, 
1)	배포 후에 다양한 언어에서 API로 추론 가능 (Python, Java, C++ 등)
2)	Tensorflow serving이 제공하는 모델 배포 기능을 활용 가능

2. Serving
Tensorflow Serving 이란 무엇인가? 모델 배포 시에 이미지 배포하는지?
Host (판정서버) 와 Client (판정대상)를 분리하여 실제 양산 환경에서 보다 효율적으로 모델을 배포 (Serving)할 수 있는 Tensorflow 공식 추천 모델 배포 방식이다. Cloud 환경에서 모델을 생성 및 배포 할 경우에 최적화 되어있는 방법으로, AWS, Google Cloud 를 비롯한 다양한 Google 내부 서비스에서 실제 모델을 배포/관리하는데 사용되고 있으며, 주로 기능은 아래와 같다.
-	Model life-cycle 관리
1)	Model rollback / AB testing / Canary
기존 모델이 양산에 있는 상태에서, 신규 모델을 배포 하여 두 개의 모델이 함께 prediction을 진행 (기존 모델을 내릴 필요가 없음) 하여 두 모델의 성능을 비교 가능 함. Ex) 기존 모델에 100%, 신규 모델에 20%의 샘플에 대해 판정 요청
	신규 모델의 성능 및 안정성을 검토 후 기존 모델을 내리고 싶을 경우 
새로 교체한 모델의 성능이 원하는 수준이 아니거나 버그가 발생할 경우, 손쉽게 원하는 버전의 기존 모델로 교체 가능 함 
명령어를 통해 모델을 바로 교체 (기존모델 -> 신규모델) 가 가능 함 
-	Remote Inference
gRPC 방식으로 client에서 host(model server)로 API 콜을 통해 판정 요청
-	Client 자유도
Client는 C++, Python, REST API 등 다양하게 API 호출이 가능 함 
-	Batching
일반적으로 Tensorflow에서 training할 경우에 데이터를 mini-batch로 만든 후 진행한다. 이는 GPU의 특성을 반영한 계산 성능을 극대화 하는 전략인데, batching을 추론에도 사용할 수 있다. 

 
TF Serving 주요 기능
Client - Host 분리 환경 구축
    - Docker container로 모델 서버(Host) 구동으로 Host-Client 분리 가능
    - API를 통한 인퍼런스로 Local server 다운 없이 지속적 인퍼런스 가능 
    - 다양한 Client 환경 (C++, Python, Rest)에서 API로 인퍼런스 가능
    - TFlite, TF 버전 변경에 따른 작업 최소화 (docker container 교체를 통한 인퍼런스 환경 변경)
모델 versioning 및 교체
    - 모델 재 학습 시에 해당 모델의 version을 지정하고, 원하는 version으로 쉽게 모델 교체 가능
    - 모델 교체 시 API에서 모델 version 이름만 바꿔주면 되므로 downtime 없이 모델 교체 가능
      (모델 교체 시 환경 재설정 및 테스트를 진행할 필요가 없음)
유연한 인퍼런스 환경 구축
    - 기존 모델이 양산에 있는 상태에서, 신규 모델을 배포 하여 두 개의 모델이 함께 성능 비교 진행 가능
      (즉, 기존 모델을 바로 내릴 필요가 없음)  
      Ex) 기존 모델에 100%, 신규 모델에 20%의 샘플에 대해 판정 요청
Batching 처리
    - 판정 데이터 Batching으로 동시에 인퍼런스 진행하여 throughput 향상 및 latency 향상
      (이미지 한장이 아닌 여러장을 묶어서 한꺼번에 인퍼런스를 처리 함)
    - 다수의 인퍼런스 요청시 동시다발적으로 받을 경우, 최적의 인퍼런스 batching을 구성하여 latency 극대화



