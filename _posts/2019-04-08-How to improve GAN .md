이글은 아래 블로그를 공부하고 쓰는 글 임
https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b

DCGAN 및 WGAN이 구현된 코드를 보면 모두 중간 (예를 들어 매 epoch 마다)에 이미지를 샘플링하는 코드가 있는 것을 볼 수 있다. 이는 GAN의 성능을 Quantitative하게 보기 어렵기 때문인 것인데, 이게 상당히 머리아픈 이슈같다. 즉, 이미지가 어떤 Quality로 생성되었는지를 화면에 뿌려지는 loss로 판단할 수가 없다는 것인데.. 그리고 낮은 loss가 진짜 좋은 quality의 이미지를 만드는지 자체도 의문인 것이 트레이닝을 어렵게 만드는 것 같다.

암튼, 위의 블로그에서 강조하는 3가지 이슈는 다음과 같다.

1. GAN 학습 시에 주로 발생하는 3가지 문제
---------------------------------------
* Non-convergence
- 모델을 학습 시킬 때 모델의 loss가 converge하지 않고, unstable해지는 경향 발생

* Mode collapse
- generator가 한가지 또는 제한된 유형의 이미지만 생성하는 현상

* Slow training
- gradient가 사라지는 현상 (loss가 잘 안줄어..)

2. 이럴 경우 생각해 볼 수 있는 해결책 
*Change the cost function for a better optimization goal.
*Add additional penalties to the cost function to enforce constraints.
*Avoid overconfidence and overfitting.
*Better ways of optimizing the model.
*Add labels.


to be continued

