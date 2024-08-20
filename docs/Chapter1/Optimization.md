# Optimization (Training)
Loss function을 정의했으면, 이를 최소한으로 하는 $\theta^\star$를 찾아야 합니다.
이 과정을 optimization, 혹은 training이라고 합니다.

## Gradient Descent
Deep learning에서는 다양한 optimizer를 사용합니다.
대부분이 gradient descent의 변형입니다.
Gradient descent는 $\nabla_\theta L(\theta)$의 값을 이용해서 $L$을 줄여나가는 방법입니다.

```{prf:definition}
:label: def-gradient-descent

Gradient Descent.
Loss function $L: \mathbb{R}^p \rightarrow \mathbb{R}$이 있을 때,
radient descent with step size $\eta$는 다음과 같습니다.
\begin{equation*}
    \theta^{n+1} = \theta^{n} - \eta \nabla_\theta L(\theta^n).
\end{equation*}
```

한 번에 minimizer를 찾는 것이 아니고, 여러 번 반복을 통해 minimizer로 다가가는 sequence를 만들어 minimizer를 찾는 방법입니다.
$n$번째 반복에서 parameter를 $\theta^n$라고 할 때, parameter를 $\theta^{n+1}$로 업데이트 하는 방법중의 하나입니다.
Gradient descent를 유도하는 방법은 간단합니다.
먼저 $\theta^n$이 있을 때, parameter를 업데이트 하는 것은 방향 $d$, 크기 $\eta$ 두가지가 필요합니다.
여기서 업데이트 크기가 step size $\eta$가 됩니다.
방향을 찾는 방법 중 하나가 gradient descent입니다.
먼저 $L$을 $\theta$에 대해 linearization 해보겠습니다.
\begin{equation*}
    L(\theta + d) \approx L(\theta) + \nabla_\theta L(\theta) \cdot d.
\end{equation*}
우리는 좌변을 $d$에 대해 minimization하고 싶습니다.
하지만 이는 어려우므로 대신 우변을 minimization 합니다.
우변은 linear하기 때문에, 우변이 줄어드는 방향을 알 수 있습니다.
1차 함수 $y = ax + b$를 생각해 보면, $a$가 양수인 경우 $x$가 감소하면 함수가 감소합니다.
즉 기울기와 반대 방향으로 움직여야 합니다.
기울기가 $\nabla_\theta L(\theta)$로 주어졌으므로 그 반대인 $d = -\nabla_\theta L(\theta)$를 선택하면,
\begin{equation*}
    L(\theta) + \nabla_\theta L(\theta) \cdot d = L(\theta) - \| \nabla_\theta L(\theta) \|_2^2 < L(\theta)
\end{equation*}
가 되어 $L$이 줄어듦을 볼 수 있습니다.
하지만 linearization은 $d$가 작을 때만 성립합니다.
따라서 $\eta$를 작게 할 필요가 있습니다.

Gradient descent는 computational cost가 저렴합니다.
Reverse mode AD를 이용해서 $\nabla_\theta L(\theta)$를 매우 저렴하게 계산할 수 있기 때문입니다.
하지만 수렴 속도가 느리다는 단점이 있습니다.
Loss function이 flat한 곳에서는 linearization이 비교적 정확하므로 $\eta$를 크게,
반대로 sharp한 곳에서는 $\eta$를 작게 가져가야 합니다.
이러한 정보를 curvature information이라고 하고, $L$을 두번 미분해서 얻을 수 있습니다.
Gradient descent는 curvature information을 고려하지 않습니다.


## Newton's method
$L(\theta + d)$를 linearization 하지 말고, quadratic approximation을 한 후 minimizing $d$를 찾으면 Newton's method가 됩니다.
```{prf:definition}
:label: def-newton-method

Newton's method.
\begin{equation*}
    \theta^{n+1} = - \eta H(\theta^n)^\dagger \nabla_\theta(\theta^n).
\end{equation*}
여기서 $H$는 Hessian matrix $[H(\theta)]_{ij} = \frac{\partial^2 L(\theta)}{\partial \theta_i \partial \theta_j}$ 입니다.
$A^\dagger$는 Moore-Penrose pseudo inverse 입니다.
그냥 matrix inverse라고 생각하셔도 무방합니다.
```

Newton's method를 유도해 보겠습니다.
\begin{equation*}
    L(\theta + d) \approx L(\theta) + \nabla_\theta L(\theta) \cdot d + \frac{1}{2}d^T H(\theta) d.
\end{equation*}
위 식은 Taylor 정리를 사용하면 쉽게 유도할 수 있습니다.
아까 했던 것처럼, 좌변을 minimization 하는 대신 우변에서 정보를 뽑아내어 $L$을 줄일 수 있는 direction $d$를 찾는게 목표입니다.
이전과 다르게, 우변이 quadratic function이고 $H(\theta)$가 positive semidefinite matrix이므로 unique minimizer가 존재합니다.
2차 함수 $q$는 $q'(x) = 0$인 점 $x$에서 extreme value를 가집니다.
마찬가지로,
\begin{equation*}
    \nabla_d \left( L(\theta) + \nabla_\theta L(\theta) \cdot d + \frac{1}{2}d^T H(\theta) d \right) = 0
\end{equation*}
인 점 $d$에서 최솟값을 가집니다.
계산을 해 보면, 이는 $d = - H(\theta)^\dagger \nabla_\theta L(\theta)$가 됩니다.
Newton's method 역시 $d$가 작을 때 quadratic approximation이 의미가 있습니다.
따라서 $\eta$ 값을 적당히 조절해 줄 필요가 있습니다.

Newton's method는 수렴 속도가 훌륭합니다.
Minimizer를 scientific notation으로 나타냈을 때,
정확한 decimal points가 iteration 한 번 증가할 때 마다 두배가 됩니다 (quadratic convergence.)
하지만 매우 큰 단점이 있습니다.
$H(\theta)^\dagger \nabla_\theta(\theta)$를 계산하는 것이 쉽지 않습니다.
\begin{equation*}
    H(\theta) d = \nabla_\theta(\theta)
\end{equation*}
를 linear system으로 만들어서 풀면 되지만,
direct method로 풀 경우 $\mathcal{O}(p^2)$ 만큼의 메모리가 필요합니다.
Neural network의 parameter 개수가 보통 크다는 것을 생각할 때, 이 메모리 비용은 꽤나 큽니다.
Matrix-free method (e.g. conjugate gradient)는 Hessian matrix를 explicit하게 만들 필요는 없지만,
condition number가 크면 부정확합니다.
그리고 matrix-free 방법 역시 $\mathcal{O}(p^3)$ operation cost가 필요합니다.


## Other Variants
다양한 방법을 통해 {eq}`PINN-Loss`를 minimize 할 수 있습니다.
보통 adam optimizer를 많이 사용합니다 {cite}`kingma2014adam`.
다른 선택으로는 L-BFGS가 있습니다 {cite}`liu1989limited`.
Adam은 잘 동작하지만 L-BFGS는 잘 동작하지 않는 경우가 있고, 그 반대의 경우도 있고, 둘 다 잘 동작하지 않는 경우가 있습니다.
따라서 다양한 optimizer의 특성을 미리 알아두고 상황에 맞게 선택할 필요가 있습니다.

Adam은 stochastic 세팅에서 잘 동작합니다.
Adaptive weight이나 stochastic collocation points sampling 같은 알고리즘을 적용할 때는 주로 adam을 먼저 사용합니다.
그 다음에 어느정도 PINN loss가 줄어들었다고 생각되면, weight과 collocation points를 모두 고정하고 L-BFGS로 추가적인 optimization을 가져가는 경우도 있습니다.

L-BFGS는 loss function의 curvature 정보를 사용하므로, 일반적으로 adam보다 퍼포먼스가 좋습니다.
하지만 stochastic 세팅에서는 잘 동작하지 않으며, 과거에 계산했던 gradient 정보를 저장해둬야 하므로 adam보다 더 많은 메모리를 사용합니다.

Parameter space에 Riemannian metric으로 Fisher matrix를 주고 이 manifold 위에서 Newton's method를 따라 PINN loss를 줄이는 방법도 있습니다 {cite}`muller2023achieving`.
그러나 이 방법은 Hessian matrix $H$에 대하여 $Hx=b$를 풀어야 하지만, $H$의 condition number가 많은 경우 크기 때문에 실용적이라고 보기 어렵습니다.

Neural network를 여러개 도입하여 순차적으로 PINN loss를 줄여서 정확성을 machine epsilon $10^{-16}$ 수준으로 달성할 수 있는 방법도 있습니다 {cite}`wang2024multi`.
Adam과 L-BFGS만 사용했기 때문에 실용적이지만, 미분방정식의 order가 높은 경우나 nonlinearity가 있는 경우에는 아직 더 연구가 필요합니다.
