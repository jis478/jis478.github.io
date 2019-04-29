---
title: "Tensorflow 2.0, 그냥 Keras를 가져다 쓰는 건가?"   (작업 중!!!!!!!!!!!)
---

1. Tensorflow 2.0에서 일어나는 변화
----------------------------------

Tensorflow 2.0에서는 eager 모드를 통한 파이썬 다운 imperative 형식으로 코딩을 진행하게 된다. 이게 1.x와 2.0의 엄청난 차이가 발생하는 포인트인데.. 그렇다면 여기서 몇가지 드는 의문점이 있다. 

1. 기존에 1.x에서 나를 그렇게 괴롭게 만들었던 graph는 모두 어디로 간거지? 
2. 2.0 부터는 Tf.keras가 메인이라는데.. 그렇다면 keras를 쓰면 되는게 아닌가? 왜 tf 2.0을 써야하지? 즉, keras랑 2.0이랑 모델 만드는거 
   똑같은거 아니야?



먼저 2에 대해서 얘기를 해보자면, GAN 코드를 공부 중에 재미있는 코드를 발견해서 여기서 정리하고자 한다.
우선 GAN의 loss를 한번 보자.


여기서 Generator를 학습시킬때 Keras 및 tf 1.x에서는 generator가 생성해낸 가짜 이미지가 discriminator의 input으로 들어가기 때문에 graph가 서로 이어지게 된다. 이렇게 될 경우에 Keras에서 꼭 해줘야 하는 것은, Discriminator 학습 부분에서는 Discriminator만 학습시키고, Generator 학습 부분에서는 Generator만 학습시키고 싶기 때문에 Generator 학습 시에는 Discriminator부분을 모두 freeze시켜야 하는 것이다.

1. Keras

