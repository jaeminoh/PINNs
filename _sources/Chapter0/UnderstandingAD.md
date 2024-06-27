## AD 이해하기
컴퓨터로 계산을 할 때, 모든 함수는 간단한 함수의 합성입니다.
예를 들어 인공신경망은 덧셈, 곱셈, 그리고 activation function이라는 "간단한" 함수들의 합성으로 이루어져 있습니다.
AD는 "간단한" 함수들의 미분을 미리 계산해두고, $f$를 미분할 때 미리 계산된 미분을 사용하는 방법입니다.

어떤 함수 $f: \mathbb{R}\rightarrow \mathbb{R}$가 있고,
이 함수는 "간단한" 함수들 {math}`f_1, \dots, f_N: \mathbb{R}\rightarrow \mathbb{R}`의 합성
\begin{equation*}
    f = f_N \circ \cdots \circ f_1
\end{equation*}
으로 표현할 수 있다고 가정하겠습니다.
만약 $f_i$들의 미분을 알고 있다면, 합성함수의 미분법 (Chain rule)을 통해서 $f'$를 계산할 수 있는 공식
```{math}
:label: chain-rule
    f' = \left(f'_N \circ f_{N-1} \circ \cdots \circ f_1\right) \cdots f'_1
```
을 얻을 수 있습니다.

### Dual numbers
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
먼저 $\epsilon^2 = 0$을 만족하는 "표현" $\epsilon$을 도입합니다.
해당 조건을 만족하는 실수나 복소수는 $0$밖에 없기 때문에, "표현"이라고 적었습니다.
Dual number는
\begin{equation*}
    p + t \epsilon, \quad p, t \in \mathbb{R}
\end{equation*}
($p$, $t$는 각각 primal, tangent의 첫 글자입니다.)
덧셈과 곱셈은 자연스럽게 정의됩니다.


