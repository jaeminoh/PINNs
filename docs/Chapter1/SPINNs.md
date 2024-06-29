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

```{bibliography}
```