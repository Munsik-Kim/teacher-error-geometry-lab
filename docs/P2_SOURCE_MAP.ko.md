# P2 변수·검증 대응

[공통 원문 대응표](claim_map.md)와 [고정 protocol](P2_PROTOCOL.md)에 연결합니다. 원문 §3.1·§3.2·§4.1·§4.2·§7.1을 읽고 대응한 과거 기록은 archive 복원 후 docs/P2_SOURCE_MAP.ko.md와 docs/P2_REPORT.ko.md에 byte 그대로 남습니다. 공개 설명 사본과 역사적 원문을 구분합니다.

|구현 변수|의미·출처|대조|
|---|---|---|
|Sigma,A,beta,delta,n,lambda|Table 1 Gaussian d5/r3/n27 anchor, lambda=.45 고정|config와 freeze hash|
|sigma2,tau2,omega|분산 1.3,.7 및 공분산 .22|noise factor가 [[1.3,.22],[.22,.7]] 재구성; determinant .8616|
|w_beta,w_gamma,C,u_beta,u_gamma|원문 Gaussian 고정 표현 투영|solve(C,A Sigma beta/gamma), 독립 좌표 분해|
|B,B_A,D_A,G_A,S_A|.2378,.08,.0753,.4625,.19|일반 행렬 경로와 scalar/Fraction 경로|
|FO,FT,RO,RM|각각 full outcome/full teacher/representation outcome/representation mixed|동일 데이터 hash, 고유 row key, 실제 fitted risk와 독립 기대값|
|RM h,cross_term|표본별 변동과 부호 있는 교차항|5차원 quadratic risk=표현 손실+편향+cross+quadratic|

원문 최적 lambda의 최소값과 이번 lambda=.45 값을 혼동하지 않습니다. NumPy root seed 2026102102는 원문 PyTorch seed 20261016과 다릅니다. 같은 분포·기대 risk의 부분 재현이며 원문 표본의 bitwise 복제가 아닙니다.
