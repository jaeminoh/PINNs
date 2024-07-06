# Neural Networks
인공신경망 (neural networks)은 affine transformation, 그리고 nonlinear activation function의 반복된 합성입니다.
Affine transformation의 횟수를 depth, affine transformation의 output dimension을 width라고 합니다.
이 함수를 fully connected neural networks (FNNs) 혹은 multi-layer perceptrons (MLPs)라고 부릅니다.

```{prf:definition}
아키텍쳐 {math}`[n_0, n_1, \dots, n_d]`, activation function {math}`\phi`가 있을 때,
MLP는 다음 함수로 정의합니다.
\begin{equation*}
    \mathrm{MLP}(x;\theta) = A^d \circ \left( \phi \circ A^{d-1} \right) \circ \cdots \circ \left( \phi \circ A^1 \right)(x).
\end{equation*}
여기서 {math}`A^i(h) = W^i h + b^i`로, {math}`W^i \in \mathbb{R}^{n_i \times n_{i-1}}` 그리고 {math}`b^i \in \mathbb{R}^{n_i}` 입니다.
또한 activation function은 component-wise하게 계산합니다.
$\theta$는 네트워크 파라미터들의 집합 {math}`\{(W^i, b^i)|i=1, \dots, d\}`을 나타냅니다.
```

```{prf:remark}
Affine transformation $A^i(h) = W^i h + b^i$는 matrix $W^i$와 vector $h$의 곱을 포함하고 있습니다.
이 matrix-vector product의 계산 복잡도는 $\mathcal{O}(n_i n_{i-1})$ 입니다.
이는 graphical processing unit (GPU)에서 굉장히 빠르게 계산할 수 있습니다.
왜 large scale deep learning에서 GPU가 꼭 필요한지,
왜 checkpointing 같은 방법에서 affine hidden layer output들을 저장하는지 등의 이유가 여기에 있습니다.
```

```{prf:remark}
현대적 deep learning에서는 더 복잡하고 거대한 구조의 neural network를 사용합니다.
하지만 본 책에서 다룰 문제들에 있어서는 MLP마저 polynomial-based 방법들보다 훨씬 model의 복잡도가 크고, 느립니다.
```


## Universal Approximation Theorem
```{prf:theorem}
:label: universal-approximation-theorem

Universal approximation theorem.

어떤 compact domain $\Omega$에서 정의된 연속함수 $f$가 있고
에러 레벨 $\epsilon$이 주어졌을 때
\begin{equation*}
    \sup_{x\in \Omega}|\mathrm{MLP}(x;\theta) - f(x)| < \epsilon
\end{equation*}
을 만족하는 $\theta$를 항상 찾을 수 있습니다.
```

```{prf:remark}
간혹 universal approximation theorem이 인공신경망이 잘 동작하는 이유가 된다는 설명을 듣곤 합니다.
하지만 이는 틀린 설명입니다.
사실 polynomial들도 universal approximation theorem을 만족합니다.
만약 위 설명이 맞다면, polynomial-based 모델들도 neural network 만큼 퍼포먼스를 내 줘야 합니다.
Universal approximation theorem은 단지 최소한의 이론적인 보장일 뿐입니다.
```