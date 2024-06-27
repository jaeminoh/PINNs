# Introduction
2017년에 arxiv에 처음 등장한 논문 physics-informed neural networks (PINNs)에서 시작된 physics-informed machine learning (PIML)은 전 세계적으로 많은 관심을 받고 있는 머신러닝 / 수치해석의 한 분야입니다.
하지만 안타깝게도 한글로 된 자료가 아직 많지 않습니다.

이 저장소의 목적은 **한글로 된 physics-informed neural networks 강의를 만드는 것**입니다.

강의의 제목은 제가 처음 머신러닝을 접하게 된 계기인 [모두를 위한 머신러닝](https://hunkim.github.io/ml/)에서 따왔습니다.


## Programming Language: Python
대부분의 머신러닝 연구는 Python, 그 중에서도 PyTorch를 기반으로 이루어집니다.
Julia와 같은 더 최근 language를 사용할 수도 있겠지만, 아무래도 사용 방법이 Python 만큼 간단하지는 않습니다.
따라서 본 강의에서는 Python을 주된 프로그래밍 언어로 사용합니다.


## Machine Learning Library: JAX
PyTorch가 머신 러닝 연구의 큰 부분을 차지하고 있지만, 적어도 PINNs에 대해서는 JAX에 비해 부족한 부분이 있습니다.
바로 forward mode automatic differentiation (AD)과 just-in-time (JIT) compilation 입니다.
PyTorch 2.0에 들어오면서 jvp 함수와 compile 함수가 생기면서 위 두 기능에 대한 PyTorch 유저의 갈증이 어느 정도 해소가 되었습니다.
그러나 아직 Taylor mode AD 같은 조금 더 고급 툴은 지원이 되지 않습니다.

학습 속도를 개선하는데 있어 forward mode AD가 상당히 중요한 역할을 하는 만큼, 본 강의에서는 머신 러닝 라이브러리로 JAX를 사용합니다.


## 목차
0. 배경 지식
    - 미분방정식
    - 인공신경망
    - Automatic Differentiation 
1. Physics-informed neural networks 소개
    - 라플라스 방정식
    - 포아송 방정식
    - 열 방정식
2. 학습이 잘 되지 않는 경우 소개
    - Advection 방정식
    - 헬름홀츠 방정식
3. 학습이 잘 되지 않는 이유에 대한 설명
    - Gradient Pathology
    - Neural Tangent Kernel
4. 학습 개선 전략들
    - Sine activation function
    - Gradient Enhanced Training (a.k.a. Sobolev Training)
    - Learning Rate Annealing
    - Causal Weighting
    - R3 Sampling
    - 그 외
5. 학습 속도 향상 전략들
    - Taylor mode AD
    - Separable PINNs


## [JAX](https://jax.readthedocs.io/en/latest/index.html) 관련 문서들
- [JAX 설치](https://jax.readthedocs.io/en/latest/installation.html)
- [Optax - Gradient Descent based Optimizers](https://optax.readthedocs.io/en/latest/)
- [JAXopt - General Optimizers](https://jaxopt.github.io/stable/index.html)


## PINNs 관련 논문들
- [Raissi et al., Physics-informed neural networks](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- [Karniadakis et al., Physics-informed machine learning](https://www.nature.com/articles/s42254-021-00314-5)
- [Sitzmann et al., SIREN](https://proceedings.neurips.cc/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Abstract.html)
- [Wang et al., Learning rate annealing](https://epubs.siam.org/doi/abs/10.1137/20M1318043)
- [Wang et al., Neural tangent kernel perspective](https://www.sciencedirect.com/science/article/pii/S002199912100663X)
- [Wang et al., Causal weighting](https://www.sciencedirect.com/science/article/pii/S0045782524000690)
- [Daw et al., R3 sampling](https://openreview.net/forum?id=Jzliv-bxZla)
- [Yu et al., Gradient enhanced PINNs](https://www.sciencedirect.com/science/article/pii/S0045782522001438)
- [Cho et al., Separable Physics-informed neural networks](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4af827e7d0b7bdae6097d44977e87534-Abstract-Conference.html)


## Etc
- 질문이 있으시다면 issue나 discussion에 작성, 혹은 메일 부탁드립니다.
- 모든 코드는 Ruff Linter를 사용합니다.