# Unsupervised Label Noise Modeling and Loss Correction

## Introduction
Noisy labels는 CNN 구조의 네트워크에 의해 쉽게 fitting 되는 것이 증명되었으며, 이는 모델의 일반화 성능을 약화시키게된다. -> (LGD 모델 tensorboard 그래프에서도 비슷한 현상이 존재 함) 처음에는 noisy label loss가 높다가 나중에는 fitting해서 loss가 낮아짐. 즉, 모델이 일부 noise에 fitting하게 되는 현상이 발생하는 것이다.

 

기존의 논문들은 이러한 noise label에 대해 주로 loss를 수정하는 것으로 대응해왔다. 예를 들어 bootstrapping loss의 경우 perpetual consistency term을 도입하는데, 이는 noisy sample로 인해 발생하는 에러를 보상해 주기 위해 network의 예측 값에 추가 보상을 하는 방법이다.
다른 방법으로는 각 class별 noise를 예측하여 class 확률을 수정하는 방법이 있으며, Curriculum learning처럼 쉬운 sample을 우선적으로 학습하는 방법도 존재 한다.

물론 noisy sample를 버릴 수도 있지만, 이는 sample도 모집단에서 나온 데이터 이기 때문에 sample 가진 중요한 data 분포에 대한 정보를 함께 버릴 수 있는 리스크가 존재한다. 

한편, mixup data augmentation을 활용할 경우 모델 구조나 loss 변경 없이 noise label에 강한 모델을 생성 할 수 있는 것도 증명되었다
본 논문은, 데이터 셋 내 아주 높은 비중의 noisy label이 존재하더라도 모델이 noisy ///////label에 fitting하는 것을 피할 수 있는 강인한 학습 방법에 대해 소개 한다. 또한 noisy sample을 버리는 것이 아닌, visual representations을 배우는데 함께 활용하는 장점이 있다. 
특히, 기존의 논문들은 대부분 별도의 clean data가 있다는 가정하에 알고리즘을 제시했지만, 본 논문에서는각 샘플의 loss를 기반으로 label noise에 대한 unsupervised model을 제시 하는 차별점이 있다. 즉, clean과 noisy sample들을 two-component (clean-noisy) Beta Mixture Model (BMM)의 sample로 보고 각각의 loss 값을 BMM에 fitting 시킴으로써 unsupervised model이 될 수 있는 것이다. 
각 모델의 가정하에서 posterior 확률분포는 dynamically weighted bootsrapping loss를 구현하는데 쓰이며, 이 loss를 구하는데 있어서 noisy sample를 활용하게 된다. 

### 본 논문이 제안하는 장점은 아래와 같은 4가지로 요약될 수 있다.
#### 1.	각 개별적 sample loss에 기반을 둔, 간단하지만 강력한 unsupervised noise label 모델링
#### 2.	unsupervised noise label model을 활용하여 각 sample loss을 보정함으로써 label noise에 overfitting을 방지할 수 있음
#### 3.	mix-up augmentation과 결합해서 모델의 robustness 향상 
#### 4.	극단적인 label noise (예: noise > 80%) 상황에서도 loss의 convergence를 이룰 수 있는 mix-up data augmentation 활용

## Leaning with label noise
** 이제 다룰 주요 내용은 아래 Eq (1)의 Categorical cross entropy loss를 어떻게 noise label을 잘 다룰 수 있는 loss로 바꿀 것인가에 대한 것임 **
 
   (yi는 noisy 하다는 가정)
  

3.1. Label noise modeling
우리는 데이터셋 D에 존재하는 noisy sample들을 발견하고, 이를 활용하여 loss를 수정하는 것에 목적이 있다. 가정이 매우 심플한데, 즉 random labels (noisy labels)는 clean labels보다 학습하는데 더욱 오래 시간이 걸리며, 즉, noisy sample들은 학습의 초반 epoch에 높은 loss를 가지게 되며, sample들의 loss의 분포만으로 이를 구별 할 수 있다는 것이다. 
 

 

일반적으로 SGD로 학습되는 CNN 구조의 네트워크는 clean samples들을 fitting할정도로 상당히 학습되지 않으면 noisy samples에서는 보통 fitting하지 않는다. 따라서 어떤 sample이 noisy한지 clean한지를 알려면 loss로부터 그 여부를 유추 가능하며, 여기서는 유추를 위해 mixture distribution model를 사용하게 된다.

 
 
일반적으로 mixture model중에서 가장 인기 있는 것은 GMM 이다. 여기서 l은 loss를 의미하며, k는 clean
또는 noisy라는 class를 의미한다. 즉, clean 및 noisy sample들의 loss 값의 분포를 두 개로 나누어서 보는것으로 이해할 수 있다. 일반적으로 위와 같은 GMM을 많이 쓰게 되는데, GMM을 우리 환경에 적용할 경우 clean sample에 대해서는 0에서 멀어지는 poor approximation을 보이게 되어 모델로 적합하지 않다.

따라서 본 논문에서는 GMM대신 BMM (Beta Mixture Model)을 사용하는데, clean 및 noisy sample들을
가지고 BMM 모델을 만들 경우, [0,1] 구간에서 상호 대칭적이고 서로 다른 방향으로 skewed 된 형태(구분이 더욱 용이한)로 분포가 나타나는 것을 볼 수 있다. 직관적으로 보아도 분포가 서로 다르게 나오며, 더군다나 실제 GMM을 적용 후에 분류 모델을 만드는 것보다 BMM을 적용 후 분류 모델을 만들 경우 극단적인 noise 환경 (CIFAR-10 noise 80%)에서 5 points 정도의 ROC-AUC 스코어가 증가하는 것을 확인 할 수 있었다.   
아래는 [0,1] 구간으로 normalized 된 l에 대한 Beta 분포 pdf를 의미하며, Beta분포를 가지고 만드는 BMM은 GMM을 대체하는 역할을 하게 된다.
 

Expectation Maximization (EM)을 활용해서 BMM을 sample들에 fitting 하게 되는데, 특히 latent variable을 도입하는데, 이는 mixture component k로부터 생성된 l의posterior 분포이다. E-step에서는  고정해놓고, Bayes rule를 이용해서 posterior 분포를 업데이트 하게 된다.한편, M-step에서는 E-step에서 추정한 를 고정 시킨 상태에서 아래와 같이  를 추정한다.
 
여기서  는 loss  의 weighted average로써,  로 표현되며,  는 weighted variance estimate로써
   로 표현된다. 

업데이트된 mixing coefficients  는 아래와 같이 계산된다. 
 

최종적으로 특정 샘플이 clean 또는 noisy라는 것에 대한 최종확률 값은
다음과 같은 posterior 분포를 통해 구할 수 있다.  
(k=0이면 clean, k=1이면 noisy 샘플임)

단, 여기서 mixture 분포를 추정하기 위해 쓰이는 loss는 전형적인 cross-entropy loss이지만, 
다음 섹션에서 보듯이 label noise를 반영하기 위해서 수정 term이 추가되는 형태로 변형되게 될 것이다.

3.2 Noise model for label correction
일반적인 cross-entropy loss는 noise label 샘플에 대해서도 모두 fitting하게 되기 때문에, 기존 cross-
entropy loss에 perceptual term을 추가한 변형된 static hard bootstrapping (예: wi는 0.2) 는 아래와 
같다. (여기서 zi(logit)를 hi (softmax) 로 바꾸면 static soft bootstrapping loss)

Before:
 
After:
 
하지만 wi를 모두 고정시킬 경우, 여전히 noise label에 대한 차별성 없이 clean 및 noise label 샘플에 대해 
fitting을 하게 되는 문제가 있기 때문에, wi는   BMM 모델을 이용해서 다이내믹하게 
fitting하도록 한다. 여기서 BMM모델은 매 epoch마다 각 샘플의 cross-entropy loss를 활용해서 추정하게 
되며, 이를 본 논문에서 사용하는 dynamic bootstrapping loss 이라 정의한다.
즉, clean샘플은 wi이 작기 때문에 1-wi값이 크게 되어 ground-truth label yi에 더욱 의존하게 되고, 반대로
noise 샘플은 zi값에 의존하게 된다. 

3.3. Joint label correction and mix up data augmentation
두 개 샘플 pair에 대해 아래와 같이 mix-up을 수행하는데, mix-up은 일반적으로 label noise에 강건한 
테크닉이다. 여기서 계수는 beta 분포에서 랜덤 추출을 하며, alpha = beta 일 경우 계수는 0.5 부근에서 주
로 추출되게 된다.
 
 
Mix-up augmentation은 clean과 noisy sample를 합치는 방법인데, 반드시 clean 과 noisy 
일 필요는 없고, 효과는 떨어지겠지만 noisy 와 noisy 샘플이라도 (우연적으로) 한 쪽 샘플의 noisy label이 
다른 샘플의 true label일 수도 있기 때문에 유용할 수 있는 방법이다.

샘플과 label이 모두 mixed 되기 때문에 모델이 noisy sample로부터 직접적으로 unstructured noise를 배
우는 것이 아닌, mixed 된 샘플로부터 structured data를 배우는 효과가 있기 때문에 학습이 noisy 샘플에 
overfitting하는 을 막을 수도 있다.

하지만 앞서 설명한대로 mix-up은 만약에 pair로 이루어진 샘플(p,q)들이 서로 noise 샘플이라면 그 효과가 떨어질 수 밖에 없는데, 따라서 우리는 mixup과 dynamic bootstrapping을 혼합한 방법으로 아래와 같은 loss을 제안한다.
Fused Dynamic Boostrapping loss with mixup
 
 
 

Experiments
 
 
 
 


 
