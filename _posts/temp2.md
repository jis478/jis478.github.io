# Network Bleding 실험

## 1. 실험 준비 

- 실험목적: 웹툰이미지를 활용해서 Network Blending을 할 경우, 기존 StyleGAN 웹툰 이미지 생성보다 정성적 성능 향상이 있는지 여부 확인 <br>
           -> FFHQ feature를 활용하여 일그러지는 얼굴처럼 인지 품질을 떨어뜨리는 비정상 이미지 최소화 여부 판단 <br>
           -> StyleGAN으로 Image to Image translation 가능 여부 판단 (다음 실험) <br>
- fine-tuning 데이터셋: 여신강림 face dataset 432 장 (256x256 resized) 
- pre-trained 모델: 
    (A) FFHQ-config-f (256x256) <br>
    (B) Anime_config-f (512x512) 
- fine-tuned 생성 모델
  StyleGAN V2 에서 제공하는 기본 하이퍼파라미터 사용. P40 2 GPUs <br>
    (C) 여신강림 (fine-tuned on (A)) : 학습시간 1d 10hrs <br>
    (D) 여신강림 (fine-tuned on (B)) : 학습시간 1d 2hrs <br>


## 2. 결과 

#### 1) Low : 여신강림 fine-tuned    high: FFHQ pre-trained (Disney case 유사)

<img1>

- 8x8에서 blending 된 경우에는 여신강림의 8x8 global structure가 남아있는 상태에서 16x16 이상의 local feature은 모두 FFHQ에서 오는 효과 
- 8x8에서 여신강림의 global structure (큰 눈)에 대해 FFHQ 모델이 "선글라스"로 매핑하는 것을 볼 수 있음. ffhq dataset에서는 비정상적으로 큰 눈이 없기 때문으로 추정
- bleding resolution 이 높아질 수록 여신강림의 local 특성을 가진채로 upscaling 생성 되기 때문에 여신강림의 texture가 도드라지게 나타 남. 
- bleding resolution 이 높아질 수록 여신갈님에서 local 특성을 만들어내기 때문에 여신강림 dataset에서 없는 선글라스는 안보이게 됨


#### 2) Low: FFHQ pre-trained    high: 여신강림 fine-tuned (Ukiyoe case 유사)

<img2>

- 8x8에서 ffhq의 global feature가 생성 (모자,측면 모습 등; 여신강림에는 없는)되며, 이를 여신강림 network가 무리하게 무리하게 매핑 시도함 
- Ukiyoe와 같이 자연스러운 이미지 생성이 어려움. 이미지의 다양성 부족, 이미지 resolution (Ukiyoe: 1024, 여신강림:256) 원인 추정 


#### 3) Low : 여신강림 fine-tuned    high: Anime pre-trained

<img3>
           
- 8x8에서 blending 된 경우에는 여선강림의 8x8 global structure가 남아있는 상태에서 16x16 이상의 local feature은 모두 Anime pre-trained 에서 오는 효과 
  - pose가 여신강림의 pose이기 떄문에 다양함 (anime dataset은 모두 정면(frontal) 이미지 임)
  - 여신강림 이미지는 모두 face zoom이 되어있어서 화면에 얼굴이 가능찬 anime 같은 이미지 생성
- bleding resolution 이 높아질 수록 여신강림 + Anime feature가 서로 혼합해서 나타나고 있음 (global은 여신강림, local은 anime) 
  (예: 32x32에서 머리 스타일은 모두 여신강림이지만, 색감은 Anime dataset임)
  
  
#### 4)  Low: Anime pre-trained    high: 여신강림 fine-tuned

- 수행 중
