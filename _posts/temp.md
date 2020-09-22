
# StyleGAN network bleding

## 1. 참고자료

- justin <https://www.justinpinkney.com/ukiyoe-yourself>
- Doron <https://twitter.com/Norod78/status/1297541204104507397/photo/1>
- gwern <->


## 2. 소개

- 두 개의 stylegan network (base, fine-tuned) 를 layer단위로 bleding 하는 기법으로써, justin pinkney가 제안 (논문이 아닌 개인 tech 블로그 개제)
- Stylegan의 경우 학습 시에 많은 GPU 리소스를 필요로 하기 때문에 custom dataset으로 from scratch로 쉽게 접근하기가 어려움, 따라서 fine-tuning을 활용하여 custom dataset에 대해 튜닝 시도
- 하지만, 단순히 fine-tuning 할 경우에 기존 base network의 weight가 유실되어 base network가 가진 feature들을 활용하기가 어려워짐
  - Ukiyoe 이미지를 fine-tuning으로 학습한다면 인물 pose가 모두 ukiyoe에 있는 측면 pose를 가진 이미지가 생성
  - 이미지 수량이 부족하므로 Fine-tuning 결과가 자연스럽지 못할 수 있는 단점 존재 (얼굴 형태 어그러짐 등; link) 
  
- 하지만, network bleding을 사용한다면 FFHQ pre-trained가 가지고 있는 정면 pose가 남은 상태에서 texture가 Ukiyoe 특징을 가진 이미지를 생성 할 수가 있음

  1) FFHQ dataset pre-trained network 준비 
  
  2) Ukiyoe dataset fine-tuned network 준비 (FFHQ dataset pre-trained network에 fine tuning) 
  
  3) Blended network 생성 
      - Low resolution decoding layer에 FFHQ pre-trained network 사용
      - High resolution decoding layer에 Ukiyoe fine-tuned network 사용
      - 예시) low resolution layer를 64로 한다면, [8,16,32,64] 까지는 FFHQ network 사용, [128, 256]은 Ukiyoe network 사용
  
  4) Blended network로 image 생성 
      - random variable 추출 -> image 생성



블랜딩 후 제너레이션

- 우키요에 (Ukiyoe) 이미지를 생성하고 싶다면 다음과 같이 수행 


- Stylegan을 이미지 generation이 아닌 translation 관점에서도 생각해 볼 수 있는데, 예를 들어 FFHQ 사람 이미지를 Ukiyoe로 변환하고 싶은 경우 



- 예를 들어 사람 (FFHQ dataset) -> 우키요에 (Ukiyoe dataset)으로 변환 하고 싶은 경우 다음과 같은 절차를 수행

  1) FFHQ dataset pre-trained network 준비
  
  2) Ukiyoe dataset fine-tuned network 준비 (FFHQ dataset pre-trained network에 fine tuning) 
  
  3) Blended network 생성 
      - Low resolution decoding layer에 FFHQ pre-trained network 사용
      - High resolution decoding layer에 Ukiyoe fine-tuned network 사용
      - 예시) low resolution layer를 64로 한다면, [8,16,32,64] 까지는 FFHQ network 사용, [128, 256]은 Ukiyoe network 사용
  
  4) Blended network로 인퍼런스 수행 
      - FFHQ 이미지 -> FFHQ pre-trained network (encoder) 로 latent feature 생성 
      - latent feature -> Blended network (decoder)로 이미지 생성

