# Teacher Error Geometry Lab

**교사 오류의 방향은 학생의 실제 정답 오차에 어떤 영향을 줄까요?** 기존 선형 교사–학생 회귀 이론을 독립 구현하고, 해석식과 실제로 학습한 학생의 risk를 대조했습니다. Gaussian 생성기, 교사 target, FP64 OLS, 모집단 risk, paired 통계와 저장 기록 재현 도구를 제공합니다. **Canonical fit 248,000행**을 보존하고 CPU로 검산할 수 있습니다.

**Munsik Kim** · [English](README.md) · [CPU workflow](https://github.com/Munsik-Kim/teacher-error-geometry-lab/actions/workflows/cpu.yml) · [Releases](https://github.com/Munsik-Kim/teacher-error-geometry-lab/releases)

![P1 표본 수별 truth risk: 이론 곡선과 기록된 평균 및 pointwise 95% 구간](recorded/p1/figures/01_truth_by_n.png)

P1의 rho=0, n=30 조건에서 visible 교사 오류는 교사 일치 MSE가 사실상 0이어도 학생 truth MSE가 **0.250000**입니다. Hidden 오류는 실측 **0.008461**, 이론 기대값 **0.008929**입니다. 교사의 답과 일치하는 정도와 실제 정답 정확도는 다른 지표입니다.

## 무엇을 재현했는가

원 논문은 Agus Sudjianto·Denis Burakov의 [*Teacher Supervision, Fixed Representations and Calibration: Exact Risk Calculations for Candidate-Scoring Decision Models*, arXiv:2610.26797v1](https://arxiv.org/html/2610.26797v1)입니다.

- **P1:** §4.2의 2차원 Gaussian 특수형을 이용한 통제 예제입니다. 84설정에서 고유 dataset 62,000개와 canonical fit 168,000개를 기록했습니다.
- **P2:** Table 1의 d=5, r=3, n=27, lambda=0.45 Gaussian 설정입니다. 네 회귀 방식이 20,000 datasets를 공유하며 fit 80,000개를 만듭니다.

| P2 방식 | 기대 truth risk | 기록된 평균 |
|---|---:|---:|
| 전체 입력 / outcome (FO) | 0.309524 | 0.310155 |
| 전체 입력 / teacher (FT) | 0.404467 | 0.403867 |
| 축소 입력 / outcome (RO) | 0.260000 | 0.259864 |
| 축소 입력 / mixed target (RM) | 0.206875 | 0.207679 |

Fitted risk와 기대 risk를 분리하고 독립 계산 경로로 대조했습니다. 기존 검증은 **36 CPU tests와 canonical fit 248,000행**을 포함합니다. [결과](docs/RESULTS.ko.md), [전체 자릿수 facts](data/results.json), [원문·코드 대응표](docs/claim_map.md)에 정밀 수치와 가정을 보존했습니다.

## 고정 Transformer 표현 부록

[선형 feature 부록](followups/linear_feature_closeout/README.ko.md)은 저장된 SmolLM2-135M feature에서 비Gaussian 입력의 선형 readout을 점검합니다. PCA16을 PCA64로 넓히자 gold-only validation MSE는 **0.300690에서 0.175955로** 개선됐습니다. 이후 독립 qualification은 교사 오류 크기가 등록된 통제 범위를 벗어나 중단했습니다.

별도 사후 분석에서는 원 오류를 그대로 두고 OLS 네 개를 학습했습니다.

| 오류 방향 | Truth risk, ± 평균 | 실제 오류 에너지당 전달률 G |
|---|---:|---:|
| High | 0.4164 | 1.0000 |
| Low | 0.1992 | 0.0619 |

한 train dataset과 재사용 qualification의 사후 기술통계이며, G는 정확도가 아닌 에너지의 비율입니다. 등록한 H1 main과 final은 미실행이고 과거 중단을 PASS로 바꾸지 않습니다.

## 두 가지 CPU 재현 경로

저장소 root에서 Python 3.11을 사용합니다. 선언된 CPU 환경을 한 번 설치하면 두 경로 모두 모델 weight·GPU·원 연구 디렉터리 없이 실행할 수 있습니다.

```bash
python3.11 -m venv .venv
.venv/bin/python -I -m pip install -r requirements-lock.txt
.venv/bin/python -I -m pip install --no-build-isolation --no-deps .
.venv/bin/python -I -m pip check
```

**원문 재현 기록:** 증거를 복원하고 P1/P2 저장 행과 선택 seed를 검산한 뒤, full sweep을 다시 학습하지 않고 summary와 그림을 만듭니다.

```bash
.venv/bin/python -I -B -m teacher_error_geometry verify-public --output /tmp/teg-public-NEW
```

**Feature 부록:** 저장 계수·prediction을 별도 QR 경로로 검산하고, 추가 fit 없이 표와 그림을 만듭니다.

```bash
.venv/bin/python -I -B followups/linear_feature_closeout/scripts/check_package.py
.venv/bin/python -I -B followups/linear_feature_closeout/scripts/closeout.py verify --run followups/linear_feature_closeout/results/closure_posthoc_01 --output /tmp/teg-feature-verify-NEW
.venv/bin/python -I -B followups/linear_feature_closeout/scripts/closeout.py summarize --run followups/linear_feature_closeout/results/closure_posthoc_01 --output /tmp/teg-feature-summary-NEW
```

출력은 매번 존재하지 않는 새 외부 경로를 사용합니다. 부록의 synthetic-array tests 9개와 기록된 수치 대조 53개는 기존 원문 재현 검사의 36 tests와 별도입니다.

## 근거 읽기

- [원문 재현 안내](docs/REPRODUCTION.md) · [이론](docs/theory.md) · [원 실행 이력](docs/HISTORY.md)
- [선형 재현 최종 보고서](followups/linear_feature_closeout/FINAL_REPORT.ko.md) · [부록 재현 안내](followups/linear_feature_closeout/REPRODUCTION.md)
- [통합 provenance·접근 이력](followups/linear_feature_closeout/PROVENANCE_AND_HISTORY.md) · [게시 이력](docs/PUBLICATION_HISTORY.md)

완료 범위는 Gaussian 회귀와 고정 pretrained feature의 선형 학생입니다. 상세 가정과 미실행 확장은 [통합 이력](followups/linear_feature_closeout/PROVENANCE_AND_HISTORY.md)에 모았습니다. NumPy/lstsq와 Matplotlib를 사용했고 AI가 구현·내부 점검을 보조했습니다. [MIT](LICENSE)는 독립 작성물에 적용하며 원문·upstream 권리는 별도입니다([NOTICE](NOTICE)).
