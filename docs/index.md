# Introduction
Physics-informed neural networks (PINNs)는 2018년 11월에 "The journal of computational physics"에 온라인 버전으로 처음 등장했습니다[@raissi2019physics].
이는 neural networks 모델에 미분방정식 관련 정보를 추가하는 방법으로, 곧이어 일반적인 머신러닝 모델에 물리적 정보를 추가하는 framework인
physics-informed machine learning (PIML)[@karniadakis2021physics]으로 일반화되었습니다.
이 연구는 전세계적으로 많은 관심을 받았습니다.
제가 PINNs 논문을 처음으로 접한 2022년 11월에, 인용 횟수는 5000회 정도 였습니다 (인용 횟수 기준으로 gPC 논문보다 아래에 있었음).
2024년 7월 3일 Google Scholar 기준으로 인용 횟수는 거의 두배가 되었습니다 (9621).
하지만 안타깝게도 한글로 된 자료가 아직 많지 않습니다.

이 책의 목적은 **한글로 된 physics-informed neural networks 강의를 만드는 것**입니다.

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


## [JAX](https://jax.readthedocs.io/en/latest/index.html) 관련 문서들
- [JAX 설치](https://jax.readthedocs.io/en/latest/installation.html)
- [Optax - Gradient Descent based Optimizers](https://optax.readthedocs.io/en/latest/)
- [JAXopt - General Optimizers](https://jaxopt.github.io/stable/index.html)


## Etc
- [email](mailto:jaeminoh.math@gmail.com)
- 어떤 종류의 PR도 환영합니다.