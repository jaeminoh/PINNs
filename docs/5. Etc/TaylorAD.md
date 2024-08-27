# Taylor Mode Automatic Differentiation
미분을 더 많이 할 수록 계산 비용이 증가합니다.
먼저 finite difference formula를 예로 들어보겠습니다.
함수 $f: \mathbb{R} \rightarrow \mathbb{R}$이 있습니다.
이 때 1계 미분은
\begin{equation*}
    f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}
\end{equation*}
로 함수 $f$를 두번 evaluate하여 미분을 approximation 할 수 있습니다.
하지만 이계 미분은
\begin{equation*}
    f''(x) \approx \frac{f(x+h) - 2f(x) + f(x-h)}{h^2}
\end{equation*}
함수 $f$를 세번 evaluate 해야 합니다.
따라서 미분을 더 많이 할 수록 함수 $f$를 더 많이 계산해야하므로, 계산 비용이 증가한다고 할 수 있습니다.

Automatic differentiation의 경우도 마찬가지입니다.
Forward mode AD로 $f'(x)$를 계산하는 것은 약 $f$ 비용의 세배입니다.
미분을 한번 더 해서 $f''(x)$를 계산하게 되면, 이는 $f'(x)$를 계산하는 비용의 세배가 되므로 $f(x)$를 계산하는 비용의 9배가 됩니다.
Finite difference formula와는 다르게 forward mode AD의 계산 비용은 미분의 횟수가 증가하게 되면 지수적으로 증가합니다.

High-order 미분이 있는 방정식의 예시를 하나 보겠습니다.
```{prf:example}
:label: example-kuramoto-sivashinsky

Kuramoto-Sivashinsky 방정식은 다음과 같습니다.
\begin{equation*}
    \partial_t u + 0.5 \partial_x (u^2) + \partial_x^2 u + \partial_x^4 u = 0.
\end{equation*}
```
$x$에 대한 미분을 4번까지 하고 있습니다.
Forward mode AD로 naïve하게 계산하면 $u$ 계산 비용의 $3^4 = 81$배가 필요합니다.

이 때 Taylor mode AD를 이용하여 high order differential을 빠르게 계산할 수 있습니다.
