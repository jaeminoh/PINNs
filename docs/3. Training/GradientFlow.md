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

### Example - Stiff Ordinary Differential Equation

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


이번에는 $y: [0, \infty) \rightarrow \mathbb{R}^p$인 경우를 생각해 보겠습니다.
### Example - Stiff System of Ordinary Differential Equations
$$
    y'(t) = - A y(t),
$$
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

[여기](#example---stiff-ordinary-differential-equation)에서 보았듯이,
forward Euler 방법은 $\Delta t < 2 / \max_i \{\lambda_i\}$라는 조건이 필요합니다.

### Example - Stiff Partial Differential Equation
Heat equation은 stiff PDE중 하나입니다.

$$
    \partial_t u = \partial_x^2 u.
$$

먼저 Fourier transform을 {eq}`heat`의 양변에 취하면 다음을 얻습니다.

$$
\partial_t \hat{u}(\xi) = (i\xi)^2 \hat{u}(\xi).
$$

따라서 forward Euler 방법은 $\Delta t < 2 / \xi^2$라는 조건이 필요합니다.
$x$ 방향으로 grid를 잘게 자를 수록 $\xi$의 값은 커집니다.
예를 들어 $x$ 방향으로 $N$개의 grid point를 도입했다면, $\max \xi = N/2$가 됩니다.
따라서 $t$ 방향은 $\Delta t = O(N^{-2})$를 만족해야 합니다.



## Gradient Flow
PINN training이 실패하는 이유를 gradient flow의 eigenvalue bias로 설명할 수 있습니다[@wang2021understanding].
간단한 아이디어입니다.
Adam과 같은 optimizer는 gradient descent 기반 방법입니다.
Gradient descent는 gradient flow를 forward Euler 방법으로 time discretization 하면 얻을 수 있습니다.
따라서 gradient flow를 분석하면, gradient descent에 대한 정보를 얻을 수 있습니다.

### Definition - Gradient Flow
Model parameter $\theta$에 대한 gradient flow는 다음과 같이 정의합니다.

$$
\frac{d\theta(t)}{dt} = - \nabla_\theta L_\mathrm{PINN}(\theta(t)).
$$


[여기](#example---stiff-system-of-ordinary-differential-equations)와 비슷하게 생겼지만,
우변이 $\theta(t)$에 대해서 linear하지 않다는 차이점이 있습니다.
따라서 linearization을 하고 나서 [여기](#example---stiff-system-of-ordinary-differential-equations)와 비슷한 analysis를 하면 insight를 얻을 수 있을 지도 모릅니다.

먼저 linearization 과정을 설명하겠습니다.
함수 $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$가 있습니다.
이 때 linearization이란,
고정된 점 $x \in \mathbb{R}^n$ 근처에서 $f$를 가장 "비슷"한 linear transformation $A_x \in \mathbb{R}^{m \times n}$를 찾는 것을 말합니다.
수학적으로 표현하면,

$$
f(x + h) = f(x) + A_x h + o(|h|)
$$
입니다.
여기서 $o(|h|)$는 $\lim_{|h|\rightarrow 0} o(|h|) / |h| = 0$를 의미합니다.
$|h|$보다 빠르게 $0$으로 간다는 뜻입니다.
이 행렬, 혹은 linear transformation $A_x$를 $f$의 $x$에서의 미분이라고 하고,
Fréchet derivative라고도 부릅니다.

자세히 보면, $|h|$가 $0$에 가까우면 $f(x+h) = f(x) + A_x h$로 쓸 수 있습니다.
$f$의 $x$ 근방에서의 움직임이 $h \mapsto A_x h$라는 linear transformation으로 approximation 된다는 뜻입니다.
만약 $f$가 미분가능하다면, $A_x = J_f(x)$ 즉 $f$의 Jacobian이 됩니다.

다시 [Gradient Flow](#definition---gradient-flow)로 돌아와 우변을 $\theta(t)$에 대해 linearization 해보겠습니다.
$\nabla_\theta L : \mathbb{R}^p \rightarrow \mathbb{R}^p$를 linearization하기 위해서는

$$
\frac{\partial \left( \nabla_\theta L\right)_i}{\partial \theta_j} = \frac{\partial^2 L}{\partial \theta_i \partial \theta_j},
$$
즉 $L$의 Hessian matrix가 필요합니다.
Linearization을 위해서,
$t_0 \le t$이고 $t$와 가까운 $t_0$를 하나 잡겠습니다.
그러면

$$
\frac{d \theta(t)}{dt} = 
- \left( \nabla_\theta L(\theta(t_0)) + H(\theta(t_0))\left(\theta(t) - \theta(t_0)\right) \right)
+ o(|\theta(t) - \theta(t_0)|)
$$
을 얻을 수 있습니다.
다시 정리하면

$$
\frac{d \theta(t)}{dt} =
- H(\theta(t_0))\theta(t) - \left( \nabla_\theta L(\theta(t_0)) - H(\theta(t_0)) \theta(t_0)\right)
+ o(|\theta(t) - \theta(t_0)|)
$$ (linearized-gradient-flow)
가 됩니다.
$o(|\theta(t) - \theta(t_0)|)$ 값은 작으므로 무시할 수 있다고 가정하겠습니다.
따라서 Gradient Descent를 할 때는 {eq}`linearized-gradient-flow`를 forward Euler 방법으로 discretization하게 됩니다.
[여기](#example---stiff-system-of-ordinary-differential-equations)의 analysis를 통해 보면,
$H(\theta(t_0))$의 eigenvalues $\lambda_i$에 대해
$\Delta t < 2 / \lambda_\mathrm{max}$를 만족해야 합니다.

여기서 optimization이 잘 되지 않는 이유를 찾을 수 있습니다.
Hessian의 spectrum이 넓으면, 다시 말해서

$$
|\frac{\lambda_\mathrm{max}}{\lambda_\mathrm{min}}| \gg 1
$$
이면, eigenvalue가 큰 방향으로는 작은 learning rate이 필요하지만 eigenvalue가 작은 방향으로는 큰 learning rate이 필요합니다.
Stability 조건 $\Delta t < 2 / \lambda_\mathrm{max}$ 때문에 큰 learning rate을 취할 수 없으므로,
optimization 속도가 필연적으로 느려지게 됩니다.

PINN loss setup으로 돌아오겠습니다.

$$
L_\mathrm{PINN}(\theta) = L_\mathrm{PDE}(\theta) + \lambda L_\mathrm{BC}(\theta)
$$
입니다.
컴퓨터로 $H_\mathrm{PDE}$와 $H_\mathrm{BC}$의 eigenvalue distribution을 계산해보면
PDE 쪽 spectrum이 훨씬 넓은 것을 발견할 수 있습니다.
따라서 PINN의 optimization을 방해하는 부분은 boundary condition 보다는 PDE loss 쪽이라고 이해할 수 있습니다.