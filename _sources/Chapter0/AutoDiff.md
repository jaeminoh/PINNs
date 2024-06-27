# Automatic Differentiation
컴퓨터로 계산을 할 때, 모든 함수는 간단한 함수의 합성입니다.
예를 들어 인공신경망은 덧셈, 곱셈, 그리고 activation function이라는 "간단한" 함수들의 합성으로 이루어져 있습니다.
Automatic Differentiation (AD)는 "간단한" 함수들의 미분을 미리 계산해두고, $f$를 미분할 때 미리 계산된 미분을 사용하는 방법입니다.

```{prf:remark}
역전파 (backpropagation) 알고리즘은 reverse-mode AD를 지칭합니다.
```

어떤 함수 $f: \mathbb{R}\rightarrow \mathbb{R}$가 있고,
이 함수는 "간단한" 함수들 {math}`f_1, \dots, f_N: \mathbb{R}\rightarrow \mathbb{R}`의 합성
```{math}
    f = f_N \circ \cdots \circ f_1
```
으로 표현할 수 있다고 하겠습니다.
만약 $f_i$들의 미분을 알고 있다면, 합성함수의 미분법 (Chain rule)을 통해서 $f'$를 계산할 수 있는 공식
```{math}
:label: chain-rule
    f' = \left(f'_N \circ f_{N-1} \circ \cdots \circ f_1\right) \cdots f'_1
```
을 얻을 수 있습니다.


(speed-forward-vs-reverse)=
## Forward mode and backward mode
AD는 크게 두 가지 방법이 있습니다.
Forward mode와 reverse mode입니다.
상황에 따라 적절한 mode를 선택해서 사용한다면, 효율적인 컴퓨팅을 할 수 있습니다.

결론부터 말하자면,
함수
```{math}
    f: \mathbb{R}^{d_\mathrm{in}} \rightarrow \mathbb{R}^{d_\mathrm{out}}
```
가 있을 때
{math}`d_\mathrm{in} < d_\mathrm{out}`인 경우 forward mode가 빠르고,
반대의 경우 {math}`d_\mathrm{in} > d_\mathrm{out}`에 reverse mode가 빠릅니다.

Deep Learning의 경우 손실 함수 (Loss function) $L$의 input은 neural network의 parameter $\theta \in \mathbb{R}^p$가 되고, output은 손실 함수의 값 $L(\theta) \in \mathbb{R}$이 됩니다.
많은 경우에 $p \gg 1$ 이므로 reverse mode가 빠릅니다.


## Dual numbers
Forward mode AD를 잘 이해하기 위해서는 dual number가 무엇인지 알 필요가 있습니다.
```{prf:definition}
:label: dual-numbers

두 실수 $p$ 그리고 $t$가 있다.
표현
\begin{equation*}
    p + t\epsilon
\end{equation*}
을 dual number라고 한다.

두 dual number의 덧셈은
\begin{equation*}
    (p_1 + t_1 \epsilon) + (p_2 + t_2 \epsilon) = (p_1 + p_2) + (t_1 + t_2)\epsilon,
\end{equation*}
로 정의하고,
곱셈은 
\begin{equation*}
    (p_1 + t_1 \epsilon) \cdot (p_2 + t_2 \epsilon) = p_1 \cdot p_2 + (p_1 \cdot t_2 + p_2 \cdot t_1)\epsilon
\end{equation*}
으로 정의한다.
```