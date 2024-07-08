# Limitations of PINNs
본 챕터에서는 현재 PINN이 가지고 있는 단점들을 설명합니다.

PINN은 크게 두 가지 카테고리의 문제점이 있습니다.
1. Training이 느립니다.
2. Training이 실패하는 경우가 꽤 많습니다.

```{prf:remark}
PINN loss를 minimization 하는 것을 보통 optimization 이라고 부르고, machine learning community에서는 training이라고 부릅니다.
```


## Slow Training
미분방정식의 해를 수치적으로 구한다는 관점에서 보면, PINN은 classical methods보다 **훨씬** 느립니다.
Classical method들은 대부분 미분방정식을 선형대수 문제로 만들고,
이미 개발된 빠르고 정확한 알고리즘들을 가져다 사용하면 되기 때문입니다.
반면 PINN은 미분방정식을 optimization 문제로 만듭니다.
Convexity, linearity 둘 다 없기 때문에 보통 해결하기 복잡한 문제입니다.

예를 들어, 이전 챕터에서 공부한 1d Poisson equation의 경우
PINN은 GPU에서 1분 남짓이 걸렸지만,
Central difference finite difference method는 1초도 걸리지 않습니다.
따라서 현재 PDE를 푸는데 PINNs를 이용하는 것은 그리 바람직하다고 볼 수 없습니다.
Super high dimension으로 가게 되면 또 할 말이 생길 수 있겠습니다만,
적어도 (6 + 1) 차원 (e.g. Boltzmann equation) 까지는 polynomial-based 방법보다 느립니다 {cite}`oh2024separable`.

그럼에도 불구하고 본 책은 PINN을 통해 미분방정식을 푸는 방법에 대해 설명하고 있습니다.
정확히는 PINN loss를 minimization 하는 방법에 대해 설명하고 있다고 보아야 합니다.
부적절한 minimization 방법을 사용해 미분방정식 정보를 machine learning 모델에 주입하게 된다면
퍼포먼스에 도움은 커녕 부정적인 영향을 끼칠 수 있기 때문입니다.
또한 PINN loss에 대해서 지식이 쌓이다 보면 classical numerical method보다 좋은 퍼포먼스를 낼 수 있는 알고리즘이 나올지도 모르는 일입니다.


## Training Failures
Optimization이 잘 되어서 (ideal) PINN loss {prf:def}`ideal-PINN-Loss`가 정확하게 $0$이 된다면 아무런 문제가 없습니다.
하지만 symbolic 계산을 하는 것이 아니라면, 어려가지 이유 때문에 정확하게 $0$을 달성할 수는 없습니다.
심지어, loss가 잘 줄어든 것 처럼 보이지만 미분방정식의 해를 잘 approximation하지 못하는 경우도 있습니다 {cite}`krishnapriyan2021characterizing`.
본 챕터에서는 이 경우를 중점적으로 설명합니다.
