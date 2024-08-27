# Practical Advices

## How to choose width and depth
인공신경망의 width, depth를 정하는 일반적인 방법은 잘 알려져 있지 않습니다.
만약 numerical solution이나 analytic solution이 있다면, 이를 인공신경망으로 regression해서 지금 사용하고 있는 width, depth가 적당한지 확인해보는게 좋습니다.
Regression이 안되는데 PINN optimization이 되는 경우는 흔치 않습니다.

이미 numerical solution이나 analytic solution이 있다면, 그것을 사용해서 미리 width와 depth를 정하는 것은 반칙 아니냐고 이야기 할 수도 있습니다.
이는 단지 일을 효율적으로 하기 위한 조언입니다.
이렇게 regression을 통해 찾은 width와 depth를 기준으로 빠르게 prototyping해서 algorithm의 퍼포먼스를 확인할 수 있습니다.
만약 논문을 작성하신다면, "numerical solution을 regression해서 width와 depth를 찾았다"라고 적으면 당연히 안되고,
width와 depth를 조절해가면서 에러가 어떻게 변하는지 조사한 table을 첨부해주면 좋습니다.
