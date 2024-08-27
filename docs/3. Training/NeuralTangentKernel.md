# Neural Tangent Kernel Theory


## Neural Tangent Kernel Perspectives
Gradient flow는 gradient descent에서 step size $\eta$를 아주 작게 했을 때 얻을 수 있었습니다.
한편, neural network의 width를 한없이 크게 늘리게 되면 Gaussian process가 된다는 연구결과가 있습니다[@lee2017deep].
이를 통해서 neural network를 이론적으로 분석하는 방법 중 하나가 neural tangent kernel theory 입니다[@jacot2018neural].
같은 analysis를 PINN에도 적용해 볼 수 있습니다.
그 결과가 여기[@wang2022and]에 정리되어 있습니다.
