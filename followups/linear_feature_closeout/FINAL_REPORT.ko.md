# 선형 이론 구현·수치 재현 최종 보고서

## 질문과 직접 구현한 것

고정된 표현을 가진 학생은 교사 오류를 얼마나 물려받을까요? 이 프로젝트는 기존 선형 교사–학생 회귀 이론을 실행하고 검산할 수 있는 재현물로 만들었습니다. Gaussian 생성기, 교사 target, FP64 OLS, fitted population risk, 기대 risk, paired 통계와 저장 기록 재현 도구를 독립 구현했습니다. P1/P2의 **canonical fit 248,000행**과 실제 pretrained 표현의 선형 메커니즘 부록이 산출물입니다.

원 논문은 Agus Sudjianto·Denis Burakov의 [*Teacher Supervision, Fixed Representations and Calibration: Exact Risk Calculations for Candidate-Scoring Decision Models*, arXiv:2610.26797v1](https://arxiv.org/html/2610.26797v1)입니다. P1은 §4.2의 2차원 특수형, P2는 Table 1의 단일 Gaussian 설정을 재현했습니다. [대응표](CLAIM_MAP.md)에 가정·수식·구현 파일·검증을 연결했습니다. 이 보고서는 읽기 쉽게 수치를 반올림했으며, [facts.json](facts.json)과 [이번 raw summary](results/closure_posthoc_01/summary.json)에 전체 자릿수를 보존했습니다.

## 모형과 독립 계산 경로

P1에서는 독립 표준정규 X1과 U로 상관된 입력 X2를 만듭니다. 실제 신호와 학생 입력은 X1입니다. 교사를 `T=(1+a)X1+bU`로 쓰면 학생 입력에서 예측 가능한 오류와 Gaussian 잔차를 나눌 수 있습니다. 고정 교사/표현, 절편 없는 무규제 OLS, 독립 동분포 test 아래

```
E_train R_truth = a² + b²/(n−2)
E_train R_teacher = b²(1 + 1/(n−2)), n>2.
```

실제로 학습한 coefficient의 risk와 이 기대값을 별도 scalar·quadratic form·solver 경로로 대조했습니다. 직접 유도에는 조건부 모멘트와 chi-square 역모멘트를 사용하며, 비퇴화 risk의 분산에는 n>4가 필요합니다. 학생이 좌표를 직접 보지 못해도 입력 상관을 통해 그 오류를 예측할 수 있으므로, 미관측 좌표와 투영 0 오류는 다릅니다.

P2는 d=5, r=3, n=27, lambda=0.45, 단위 입력 covariance와 세 좌표를 남기는 고정 표현을 사용합니다. 전체/축소 입력과 outcome/teacher/mixed target의 네 방식을 학습하며, target noise의 공분산과 표현 손실을 유지합니다. 주 지표는 관측 잡음을 제외한 참 신호 risk입니다. 일반 투영 계산에는 `solve`, OLS에는 least squares를 사용했습니다. 독립 좌표·Fraction 계산으로 기대식을 확인하고, 실제 fit의 risk 분해에서는 부호 있는 교차항을 보존했습니다.

## P1과 P2의 결과

| P1 조건·지표 | 이론 기대값 | 기록된 평균 |
|---|---:|---:|
| rho0 / n30, visible truth risk | 0.250000 | 0.250000 |
| 같은 조건 teacher agreement | 0 | 약 1.5e−31 |
| rho0 / n30, hidden truth risk | 0.008929 | 0.008461 |
| rho0.9 / n1000, hidden truth risk | 0.202548 | 0.202550 |

P1은 84설정×2,000반복, canonical fit 168,000행과 고유 dataset 62,000개입니다. 여기서 teacher agreement는 학습 residual이 아닌 독립 test의 모집단 risk입니다. Visible 오류 예제는 교사와 잘 일치한다는 사실만으로 실제 정답 정확도를 보장할 수 없음을 보여줍니다.

| P2 방식 | 입력 / target | 기대 truth risk | 기록된 평균 | MC-SE |
|---|---|---:|---:|---:|
| FO | 전체 / outcome | 0.309524 | 0.310155 | 0.001637 |
| FT | 전체 / teacher | 0.404467 | 0.403867 | 0.001518 |
| RO | 축소 / outcome | 0.260000 | 0.259864 | 0.001153 |
| RM | 축소 / mixed | 0.206875 | 0.207679 | 0.000791 |

네 방식이 동일한 20,000 training datasets를 공유하며 fit 80,000행을 만듭니다. RM의 가장 낮은 관측 평균은 이 고정 설정에 관한 결과입니다. 기존 verifier와 hosted CPU CI는 36 tests와 canonical 248,000행을 검산했습니다. 이번 closeout은 이 receipt와 identity를 재사용했고 full sweep을 반복하지 않았습니다. MC 구간과 4SE 조사 규칙은 이력에 보존하며 수치 계산을 정리의 증명으로 쓰지 않습니다.

## 실제 pretrained feature 부록과 이번 사후 측정

단일 SmolLM2-135M checkpoint를 동결하고 FP32 final-layer 마지막 non-padding token 표현에서 선형 head를 학습했습니다. PCA와 scaling은 calibration feature에서만 정했습니다. 과거 PCA16 task 평가는 중단됐고, 같은 cache의 사후 진단에서 고정된 후보 공간을 비교했습니다.

| OLS 공간 | Train-only CV MSE | 개발 validation MSE |
|---|---:|---:|
| PCA16 | 0.299546 | 0.300690 |
| PCA64 | 0.182789 | 0.175955 |
| Full hidden | 0.048071 | 0.044506 |

이 train sample과 개발 validation에서는 PCA16 병목을 지지하는 근거입니다. PCA가 보존하는 입력 분산과 정답 예측력은 같지 않습니다. Full-hidden의 표본 내 포화 projection을 독립 평가 예측력으로 해석하지 않았고, 고정 cal→val probe에서 기존 low-error의 일반화 개선은 관측하지 못했습니다.

후속 PCA64 실험의 독립 qualification은 gold MSE 0.183002, projection 차이 0.931817이었지만 high teacher MSE 0.233394가 등록 범위 0.2375–0.2625를 벗어났습니다. Low teacher MSE는 0.262229였습니다. Main/final을 실행하지 않고 중단했으며, 원 F2 task stop과 F2R qualification stop은 유지합니다.

이번 closeout에서는 이미 접근한 qualification을 사후 평가 자료로 재사용했습니다. 같은 PCA64 train에 `y*±e_high`, `y*±e_low` 네 target을 실제로 학습했습니다. 원 오류 함수와 gold coefficient는 고정했고, 한 train dataset의 train/eval 각 2,048행을 사용했습니다. 추가 네 OLS fit은 모두 유효하며 heldout 오류를 같게 맞추기 위한 재정규화는 하지 않았습니다.

| 오류 방향 | Truth risk R, ± 평균 | Agreement A, ± 평균 | 실제 오류 E | Risk 증가 Delta | 전달률 G |
|---|---:|---:|---:|---:|---:|
| High | 0.4164 | 0.1830 | 0.2334 | 0.2334 | 1.0000 |
| Low | 0.1992 | 0.4456 | 0.2622 | 0.0162 | 0.0619 |

*한 train dataset과 재사용 qualification의 사후 측정입니다. Gold truth risk는 0.1830이고, R/A는 실제 plus/minus fit의 평균입니다. 개별 arm의 정밀 수치와 signed cross term은 저장 기록에 있습니다.*

`Delta=(R_plus+R_minus)/2−R_gold`, `G=Delta/mean(e_eval²)`입니다. G는 실제 평가 오류 에너지를 분모로 한 ratio of means이며 정확도·성공확률·모집단 투영 비중이 아닙니다. 명목 0.25로 나누지 않았고 새로운 통계적 검정이나 여러 train seed의 신뢰구간을 만들지 않았습니다.

고정 선형 learner에서 `a=f_gold−y*`, `d=f_arm−f_gold`로 두면

```
R_arm−R_gold = mean(d²) + 2 mean(a*d).
f_plus/minus = f_gold +/- K e_train
Delta = mean((K e_train)²).
```

개별 fit의 교차항은 그대로 남기고 sign 평균에서만 소거합니다. High 오류는 고정 PCA64 선형 span에 있고 train design이 full rank이므로 G_high≈1은 알려진 OLS 선형성의 검산입니다. Low는 calibration 잔차이며 모집단 직교·독립을 뜻하지 않습니다. 그 전달률은 실제로 측정했습니다. 등록된 동일 오류 크기 H1의 효과·구간과 H2/H3는 미실행이며, 이번 G가 그 결과를 대신하지 않습니다.

## 재현과 근거 확인

부록에는 작은 PCA64 입력, label/error, IDs, 원 metadata, fitted coefficient와 행별 prediction이 들어 있습니다. Weight·GPU 없이 실행합니다. [REPRODUCTION.md](REPRODUCTION.md)에 압축을 푼 root에서의 정확한 설치·검산·fit 없는 재집계 명령이 있습니다. [원문 P1/P2 저장 기록 재현](https://github.com/Munsik-Kim/teacher-error-geometry-lab/blob/v0.1.0/docs/REPRODUCTION.md)은 별도 경로입니다.

Primary target 네 개는 FP64 SVD/lstsq로 계산했습니다. 별도 reduced-QR gold/error-only 계산, 저장 coefficient의 prediction 재계산과 residual 대 quadratic MSE 대조는 53 checks PASS입니다. 독립 synthetic-array tests 9개도 통과했습니다. 허용오차는 이 결과를 보기 전에 정했고, 재집계는 fit0입니다. 정확한 호출 수·비용·source identity·과거 버그 수정·test 노출·보존 범위는 [PROVENANCE_AND_HISTORY.md](PROVENANCE_AND_HISTORY.md)와 [facts.json](facts.json)에 모았습니다.

완료 범위는 Gaussian 선형 이론의 부분 재현과 고정 pretrained feature의 선형 메커니즘 부록입니다. AI가 구현과 내부 점검을 보조했으며 사람의 독립 심사와 구분합니다. 가정·미실행 확장·원 중단 이력은 provenance 문서에 모아 보존했습니다.
