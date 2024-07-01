# Physics-informed loss
편미분방정식이 다음과 같이 있다고 하겠습니다.
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
Physics-informed loss $\mathcal{L}_\mathrm{PINN}(\theta)$는 다음과 같이 정의한다.
\begin{equation*}
    \mathcal{L}_\mathrm{PINN}(\theta) = \int_\Omega \left( \mathcal{D}[u_\theta](x) - f(x) \right)^2 dx + \lambda \int_{\partial\Omega} \left( \mathcal{B}[u_\theta](x) - g(x) \right)^2 d\sigma(x)
\end{equation*}
```

이 때 physics-informed neural networks는 다음을 만족하는 network parameter $\theta^\star$를 찾는 방법입니다.
```{math}
    \theta^\star = \arg \min_{\theta} \mathcal{L}_\mathrm{PINN}(\theta)
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
단순하게 $x_i$를 random distribution에서 뽑고, 그 점들에서 integrand를 evaluation한 값들에 평균을 취하면 됩니다.
사실 domain의 크기만큼을 평균낸 값에 곱해줘야 하지만, 이는 현재 크게 중요하지 않으므로 넘어가겠습니다.

정리하면, (empirical) PINN loss는 다음과 같습니다.
\begin{equation*}
    L_\mathrm{PINN}(\theta) = \frac{1}{N_r}\sum_{i=1}^{N_r} \left( \mathcal{D}[u_\theta](x_{i,r}) - f(x_{i,r}) \right)^2 + \frac{\lambda}{N_{b}}\sum_{j=1}^{N_b} \left( \mathcal{B}[u_\theta](x_{j,b}) - g(x_{j,b})\right)^2.
\end{equation*}
이를 minimize 하는 방법은 다양하지만, 보통 adam optimizer를 많이 사용합니다 {cite}`kingma2014adam`.
다른 선택으로는 L-BFGS가 있습니다 {cite}`liu1989limited`.