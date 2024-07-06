# Failures

PINN은 결국 neural network를 통해 PINN Loss {eq}`PINN-Loss`를 줄여서 neural network에 미분방정식 정보를 주입하는 방법입니다.
하지만 실제로 PINN을 통해 미분방정식을 풀어보려고 시도하면, 생각보다 잘 되지 않습니다.
여기에는 크게 두가지 이유가 있습니다.
1. model의 expressibility 부족 문제
2. Optimization 문제


## Expressibility Issues
만약 미분방정식의 해가 엄청 복잡한 함수라면, 보통 model의 parameter 개수를 충분히 많도록 해 주어야 합니다.
충분하지 않다면, {prf:ref}`universal-approximation-theorem`을 만족하지 못하고, model이 해를 잘 approximation 할 수 있다는 이론적 보장이 깨지게 됩니다.


## Optimization Issues
PINN loss의 minimum을 찾는 일은 간단하지 않습니다.
Neural network를 model로 사용할 경우에 PINN loss는 non-convex objective function입니다.
이 경우 gradient descent가 global minimum으로 수렴한다는 이론적인 보장은 없습니다.


### Gradient Pathologies
PINN training이 실패하는 이유를 gradient flow의 eigenvalue bias로 설명할 수 있습니다 {cite}`wang2021understanding`.
간단한 아이디어입니다.
Adam과 같은 optimizer는 gradient descent 기반 방법입니다.
Gradient descent는 gradient flow를 forward Euler 방법으로 time discretization 하면 얻을 수 있습니다.
따라서 gradient flow를 분석하면, gradient descent에 대한 정보를 얻을 수 있습니다.

(def:gradient-flow)=
```{prf:definition}
Model parameter $\theta$에 대한 gradient flow는 다음과 같이 정의합니다.
\begin{equation*}
    \frac{d\theta(t)}{dt} = - \nabla_\theta L_\mathrm{PINN}(\theta(t)).
\end{equation*}
```


### Neural Tangent Kernel Perspectives
Gradient flow는 gradient descent에서 step size $\eta$를 아주 작게 했을 때 얻을 수 있었습니다.
한편, neural network의 width를 한없이 크게 늘리게 되면 Gaussian process가 된다는 연구결과가 있습니다 {cite}`lee2017deep`.
이를 통해서 neural network를 이론적으로 분석하는 방법 중 하나가 neural tangent kernel theory 입니다 {cite}`jacot2018neural`.
같은 analysis를 PINN에도 적용해 볼 수 있습니다.
그 결과가 {cite}`wang2022and`에 정리되어 있습니다.
