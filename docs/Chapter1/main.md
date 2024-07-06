# Physics-informed neural networks
Phyics-informed neural networks (PINNs)는 데이터 기반 머신러닝 모델이 알고있는 물리법칙 (주로 미분방정식으로 기술)을 따르도록 하는 방법입니다.
방법은 아주 간단합니다.
머신러닝 모델에 parameter가 sparse하다는 추가적인 제약 조건을 걸기 위해서는 주로 Lasso penalty를 고려합니다 {cite}`tibshirani1996regression`.
비슷하게, 머신러닝 모델이 미분방정식을 만족해야 한다면, 미분방정식을 만족하도록 하는 penalty 함수를 고안해서 objective function에 더해주면 됩니다.
이 penalty를 Physics-informed loss 혹은 PINN loss라고 부릅니다.

흥미로운 점은 data가 없을 때 PINN loss를 minimize 하는 모델은 미분방정식의 해가 된다는 것입니다.
본 책에서는 data가 없는 경우를 가정하고 PINN loss를 minimize하는 방법에 대해서만 다룹니다.
만약 분석하고 싶은 data가 있는 경우에는, 사용하던 모델에다 PINN loss를 추가하여 분석함으로써 model이 물리법칙을 따르도록 유도할 수 있습니다.


## Physics-informed loss
다음과 같은 편미분방정식이 있습니다.
\begin{align}
    \mathcal{D}[u](x) & = f(x), \quad x \in \Omega, \\
    \mathcal{B}[u](x) & = g(x), \quad x \in \partial \Omega.
\end{align}
여기서 $u$는 편미분방정식의 해,
$\mathcal{D}$은 differential operator,
$\sigma$는 surface measure,
그리고 $\mathcal{B}$는 boundary condition을 나타냅니다.
```{prf:remark}
:label: remark-EvolutionaryEquations
Evolutionary equation의 경우 $t$를 $x$에 포함시켜 $\mathcal{B}$가 initial condition도 나타내게 할 수 있습니다.
```

Physics-informed loss를 정의하겠습니다.
```{prf:definition}
:label: ideal-PINN-Loss
Physics-informed loss $\mathcal{L}_\mathrm{PINN}(\theta)$는 다음과 같이 정의한다.
\begin{equation*}
    \mathcal{L}_\mathrm{PINN}(\theta) = \int_\Omega \left( \mathcal{D}[u_\theta](x) - f(x) \right)^2 dx + \lambda \int_{\partial\Omega} \left( \mathcal{B}[u_\theta](x) - g(x) \right)^2 d\sigma(x).
\end{equation*}
```

이 때 physics-informed neural networks는 다음을 만족하는 network parameter $\theta^\star$를 찾는 방법입니다.
```{math}
    \theta^\star = \arg \min_{\theta} \mathcal{L}_\mathrm{PINN}(\theta)
```

```{prf:remark}
만약 {math}`\mathcal{L}_\mathrm{PINN}(\theta^\star) = 0`이라면 {math}`u_{\theta^\star}`는 almost everywhere 미분방정식을 만족하게 되므로 미분방정식의 해가 됩니다.
```

하지만, $\mathcal{L}$은 적분을 통해 정의되어 있기에 값을 정확하게 구하는 것은 쉽지 않습니다.
따라서 quadrature rule을 통해 적분 값을 approximation하게 됩니다.
```{prf:definition}
Quadrature rule은 다음과 같습니다.
\begin{equation*}
    \int_a^b f(x) dx \approx \sum_{i=1}^N w_i f(x_i).
\end{equation*}
```
PINN 분야에서 가장 흔하게 쓰는 quadrature rule은 Monte-Carlo 입니다.
Collocation points $x_i$를 random distribution에서 뽑고, 그 점들에서 integrand를 evaluation한 값들에 평균을 취하면 됩니다.
사실 domain의 크기만큼을 평균낸 값에 곱해줘야 하지만, 이는 현재 크게 중요하지 않으므로 넘어가겠습니다.

정리하면, (empirical) PINN loss는 다음과 같습니다.
```{math}
:label: PINN-Loss
L_\mathrm{PINN}(\theta) = \frac{1}{N_r}\sum_{i=1}^{N_r} \left( \mathcal{D}[u_\theta](x_{i,r}) - f(x_{i,r}) \right)^2 + \frac{\lambda}{N_{b}}\sum_{j=1}^{N_b} \left( \mathcal{B}[u_\theta](x_{j,b}) - g(x_{j,b})\right)^2.
```

## Optimization
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
