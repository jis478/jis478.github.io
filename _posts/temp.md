
# StyleGAN network bleding


## 1. 참고자료

<https://www.justinpinkney.com/ukiyoe-yourself>
<https://www.justinpinkney.com/toonify-yourself/>
<https://twitter.com/Norod78/status/1297541204104507397/photo/1>


## 2. 소개

- 두 개의 stylegan network (base, fine-tuned) 를 layer단위로 bleding 하는 기법으로써, justin pinkney가 제안 (논문이 아닌 개인 tech 블로그 개제)
- Stylegan의 경우 서로 다른 layer가 다른 level의 feature를 생성한다는 것에 착안 함. 
  예시) Low resolution layer는 global structure 생성 (pose 등)
- Stylegan의 경우 학습 시에 많은 GPU 리소스를 필요로 하기 때문에 custom dataset으로 from scratch로 쉽게 접근하기가 어려움, 따라서 fine-tuning을 활용하여 custom dataset에 대해 튜닝 시도
- 하지만, 단순히 fine-tuning 할 경우에 기존 base network의 weight가 유실되어 base network가 가진 feature들을 활용하기가 어려워짐
  - Ukiyoe 이미지를 fine-tuning으로 학습한다면 인물 pose가 모두 ukiyoe에 있는 측면 pose를 가진 이미지가 생성
  - 이미지 수량이 부족하므로 Fine-tuning 결과가 자연스럽지 못할 수 있는 단점 존재 (얼굴 형태 어그러짐 등; link) 





#### Approch #1 이미지 생성

 하지만, network bleding을 사용한다면 FFHQ pre-trained가 가지고 있는 정면 pose가 남은 상태에서 texture가 Ukiyoe 특징을 가진 이미지를 생성 할 수가 있음

  1) FFHQ dataset pre-trained network 준비 
  
  2) Ukiyoe dataset fine-tuned network 준비 (FFHQ dataset pre-trained network에 fine tuning) 
  
  3) Blended network 생성 
      - Low resolution decoding layer에 FFHQ pre-trained network 사용
      - High resolution decoding layer에 Ukiyoe fine-tuned network 사용
      - 예시) low resolution layer를 64로 한다면, [8,16,32,64] 까지는 FFHQ network 사용, [128, 256]은 Ukiyoe network 사용
  
  4) Blended network로 이미지 생성 
      - random variable 추출 -> Blended network (decoder)로 이미지 생성




#### Approch #2 이미지 변환

- Approach #1을 조금 변형하면 Stylegan을 이미지 generation이 아닌 translation으로도 활용 가능함. 예를 들어 FFHQ 사람 이미지를 Ukiyoe로 변환하고 싶은 경우, 마지막 이미지 생성 단계를 아래와 같이 수정 할 수 있음.

  4) Blended network로 인퍼런스 수행 
      - FFHQ 이미지 -> FFHQ pre-trained network (encoder) 로 latent feature 생성 
      - latent feature -> Blended network (decoder)로 이미지 생성





## 3. Case Study


#### 3.1 Ukiyoe
- 대략 수천장의 이미지를 사용했다고 함
- Low resolution newtork가 FFHQ pre-trained 모델, High resolution network가 Ukiyoe fine-tuned 모델 사용
- 따라서 pose 같이 중요한 global structure를 FFHQ 모델로 feature를 생성, texture는 Ukiyoe 모델로 생성 하여 입히는 방식 임
- 만약 blending network를 쓰지 않고 순수 Ukiyoe fine-tuning 모델로 이미지 생성한다면, fine-tuning 학습 시에 FFHQ 모델의 global structure filter가 변형되어 pose를 캐치 못하는
 현상이 발생하는 것 확인


#### 3.2 Disney Toonification 
- Disney 캐릭터 이미지 317장 <https://twitter.com/Buntworthy/status/1297976798236598274/photo/1> 으로 network blending을 시도 함
- Low resoultion network는 디즈니 캐릭터 fine-tuned 모델, High resolution network는 FFHQ pre-trained 모델을 사용 함
- 따라서, 디즈니 캐릭터의 특징 (큰 눈, 과장된 얼굴 형태 등)이 보존된 상태에서 FFHQ 특성 (사람 피부 등 디테일)이 반영되는 것을 기대 함
  (Ukiyoe case에서는 반대로 Low resolution newtork가 FFHQ pre-trained 모델, High resolution network가 Ukiyoe fine-tuned 모델 이었음)
- 적은 수량의 이미지로도 좋은 품질을 보이고 있음. 기존 FFHQ pre-trained 모델이 가지고 있는 feature (눈, 코, 입 등)이 Disney 이미지에 잘 mapping 되면서 
  fine-tuning 확인


