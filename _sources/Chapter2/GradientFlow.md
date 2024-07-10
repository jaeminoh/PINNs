# Gradient Flow Pathologies

## Stiff Equations
Time dependent 미분 방정식은 일반적으로 다음 형태를 가집니다.

$$
    \frac{d}{dt}y(t) = f(t, y(t)), \quad y(0) = y_0.
$$

이와 같은 방정식은 먼저 time step을 잘게 쪼갠 다음,
time grid 위에서 방정식을 discretize 하고,
적당한 update rule을 따라서 discretized solution을 구하게 됩니다.
수 없이 많은 discretization 방법이 있지만, 그 중에서 가장 간단한 forward Euler 방법만을 살펴보겠습니다.

$$
    \frac{y^{n+1} - y^n}{\Delta t} = f(t^n, y^n).
$$
우변이 $y^{n+1}$에 depend하지 않습니다.
이런 방법을 explicit method라고 합니다.
만약 우변의 $y^n$을 $y^{n+1}$로 바꾼다면, 이는 implicit methods 중의 하나인 backward Euler 방법이 됩니다.

미분방정식이 stiff하다는 것의 의미는, explicit method가 잘 작동하지 않는다는 뜻입니다.
다음 예를 보겠습니다.
```{prf:example} Stiff Ordinary Differential Equation
:label: example-ode-1

$$
    y'(t) = -a y(t), \quad a > 0, \quad t \ge 0, \quad y(0) = y_0.
$$

이 방정식의 해는 $y(t) = y_0 e^{-at}$입니다.
해를 이미 알고 있지만, stiffness가 무엇인지 설명하기 위해 이를 forward Euler 방법으로 푼다고 생각해보겠습니다.
Forward Euler update rule에 해당되는 점화식 (recurrence relation)은 다음과 같습니다.

$$
    y^{n+1} = (1 - a \Delta t)y^n.
$$

만약 $|1 - a \Delta t| \ge 1$이라면, 수열 {math}`\{ y^n \}`은 발산합니다.
따라서 $|1 - a \Delta t | < 1$이란 조건이 필요합니다.
정리하면

$$
    \Delta t < \frac{2}{a}
$$
입니다.
따라서 $a>0$가 커질수록, 더 작은 $\Delta t$가 필요하게 됩니다.
반면 backward Euler 방법은 이러한 제약조건이 없습니다.
```

이번에는 $y: [0, \infty) \rightarrow \mathbb{R}^p$인 경우를 생각해 보겠습니다.
```{prf:example} Stiff System of Ordinary Differential Equations
:label: example-ode-p

$$
    y'(t) = - A y(t),
$$ (ode-p)
$A$는 symmetric positive definite matrix라고 가정하겠습니다.
($a>0$의 generalization.)

Eigenvalue decomposition을 통해 $A = Q \Lambda Q^*$로 쓸 수 있습니다.
$A$가 symmetric 하기 때문에 $Q$가 orthogonal matrix가 되고 $Q^* = Q^{-1}$,
그리고 positive definite이기 때문에 모든 eigenvalue가 양수입니다.
따라서 {eq}`ode-p`는 다음과 같이 다시 쓸 수 있습니다.

$$
    \frac{d}{dt}Q^* y(t) = -\Lambda Q^* y(t).
$$

여기서 $z(t) = Q^* y(t)$라고 하면 위 미분방정식은 decouple 된 $p$개의 미분방정식
$$
    z_i'(t) = -\lambda_i z_i(t)
$$
가 됩니다.

{prf:ref}`example-ode-1`에서 보았듯이,
forward Euler 방법은 {math}`\Delta t < 2 / \max_i \{\lambda_i\}`라는 조건이 필요합니다.
```

```{prf:example} Stiff Partial Differential Equation
:label: example-stiff-pde

Heat equation은 stiff PDE중 하나입니다.

$$
    \partial_t u = \partial_x^2 u.
$$ (heat)

먼저 Fourier transform을 {eq}`heat`의 양변에 취하면 다음을 얻습니다.

$$
\partial_t \hat{u}(\xi) = (i\xi)^2 \hat{u}(\xi).
$$

따라서 forward Euler 방법은 $\Delta t < 2 / \xi^2$라는 조건이 필요합니다.
$x$ 방향으로 grid를 잘게 자를 수록 $\xi$의 값은 커집니다.
예를 들어 $x$ 방향으로 $N$개의 grid point를 도입했다면, $\max \xi = N/2$가 됩니다.
따라서 $t$ 방향은 $\Delta t = O(N{-2})$를 만족해야 합니다.
```


## Gradient Flow
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