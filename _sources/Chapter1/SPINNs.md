# Separable Physics-informed Neural Networks
함수 $f: \mathbb{R}^d \rightarrow \mathbb{R}$이 있을 때,
지금까지는
\begin{equation*}
    f(x_1, \dots, x_d) \approx \mathrm{MLP}(x_1, \dots, x_d ; \theta)
\end{equation*}
로 함수를 approximation 해 왔었습니다.

이 형태 대신
\begin{equation*}
    f(x_1, \dots, x_d) \approx \sum_{r=1}^R \otimes_{i=1}^d \mathrm{MLP}(x_i; \theta_i)
\end{equation*}
의 형태로 PDE의 solution을 approximation 하는 방법을 separable physics-informed neural networks 라고 부릅니다 {cite}`cho2024separable`.

그냥 봐서는 어떻게 speed-up이 있는지 감이 오지 않을 수도 있습니다.
하지만 저 형태는 rectilinear grid 위에서 vectorize 하여 계산하게 되면,
forward pass의 횟수가 $O(N^d)$에서 $O(dN)$으로 줄어들게 됩니다.
높은 dimension의 rectilinear grid에서 매우 빠른 계산 속도를 보여주게 됩니다.
이는 singular value decomposition, 혹은 canonical polyadic decomposition으로도 잘 알려져 있는 방법입니다.

```{prf:remark}
Tensor neural networks (TNNs)라는 이름으로도 알려져 있으나 {cite}`wang2022tensor`, arXiv에 등장한 날짜 기준으로 SPINNs가 몇개월 정도 빠릅니다.
또한 TNN은 CANDECOMP의 형태만 사용함에도 불구하고 "tensor"라는 일반적인 이름을 붙였습니다.
Tensor method에는 여러가지 방법이 있습니다. CANDECOMP, Tucker format, tensor train format 등.
```

```{bibliography}
```