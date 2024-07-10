# Differential Equation
$I = [-1, 1]$위에 함수 $T_0(x)$가 주어져 있습니다.
{math}`T_0(-1) = T_0(1) = 0`, 그리고 $x \ne 0$에 대해 {math}`T_0(x) > 0`를 만족한다고 하겠습니다.
이는 길이가 $2$인 쇠막대기의 온도를 함수로 표현한 것으로 볼 수 있습니다.
Heat equation (열 방정식)은 시간이 지남에 따라 쇠막대기 온도가 어떻게 변하는지를 서술합니다.

```{prf:definition}
:label: def-heat-equation

Heat equation은 다음과 같이 정의합니다.
\begin{equation*}
    \frac{\partial T}{\partial t}(x, t) = \frac{\partial^2 T}{\partial x^2}(x, t) \quad (x,t) \in I \times (0, \infty).
\end{equation*}
```

만약 $T(-1,t) = T(1, t) = 0$라는 boundary condition, 그리고 $T(x, 0) = T_0(x)$라는 조건을 고려한다면,
위 열 방정식은 쇠 막대기 끝의 온도가 $0$으로 고정, 외부와 온도를 주고받지 않으며, 시간이 $0$일 때 온도의 분포가 $T_0(x)$로 주어진 쇠 막대기의 $t$초 후 온도 분포를 예측하는 방정식이 됩니다.
좌변에는 온도 분포의 시간에 따른 변화가, 그리고 우변에는 온도 분포의 공간에 따른 미분이 있습니다.
이 등식을 미분방정식 (편미분이 있으므로 편미분방정식)이라고 하고, 미분방정식을 만족하는 $T$를 미분방정식의 해라고 합니다.