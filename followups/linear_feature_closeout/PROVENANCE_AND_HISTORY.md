# 보존·접근·수정 이력과 검증 수준

이 문서는 선형 재현 단계의 독립 구현, 실제 반복 관측, 사후 진단과 미실행 범위를 한곳에 연결한다. 원 기록과 실패를 수정하지 않고 새 closeout 부록에서 설명한다.

## 선형 이론 구현부터 공개 재현물까지

P1은 원문 §4.2의 2차원 Gaussian 고정 표현 특수형을 생성·교사·solver·fitted population risk·별도 expectation·통계·그림으로 나눠 독립 구현했다. 84설정×2,000=168,000 canonical fit rows, 62,000 unique dataset keys다. 최초 paired teacher-risk 진단은 C/rho0/θπ4↔3π4의 구조적 결정성을 누락해 tiny SE로 나눴다. 결과를 관측한 뒤 분류를 고쳤고 같은 seed/표본으로 두 번째 full을 실행했다. 336,000 historical fit calls는 4,000 독립 반복이 아니다. 최초 source·판정·raw와 수정시점은 [원 이력](https://github.com/Munsik-Kim/teacher-error-geometry-lab/blob/fd8bd773e2bfd0ec9da15fbe8d1447b8b0dff343/docs/HISTORY.md)에 남아 있다.

P2는 Table 1의 d5/r3/n27/λ.45 한 설정만 구현했다. 동일 20,000 datasets에서 FO/FT/RO/RM을 학습해 80,000 rows를 만들었다. 원문 PyTorch seed와 NumPy 난수열은 다르며 기대 risk의 재현이지 원 저자 표본의 bitwise 복제는 아니다. RM의 표본 signed cross mean은 +.00041106984100782904이며0으로 강제하지 않았다. 원 P1 보존340파일 receipt도 유지된다.

v0.1.0 release commit은 `fd8bd773e2bfd0ec9da15fbe8d1447b8b0dff343`이다. P1/P2 원 실행 commit은 새 공개 Git history의 조상이 아니며 frozen source bytes로 각각 확인한다. 독립 작성물의 MIT는 원 논문·upstream 권리를 덮지 않는다. 저자 코드를 열람·복사한 이력은 없다.

[최종 release CPU CI](https://github.com/Munsik-Kim/teacher-error-geometry-lab/actions/runs/39572240567)는 36 tests와 248,000 canonical rows를 검산한 과거 성공 기록이다. 첫 공개 CI 실패 및 이후 다른 CPU에서 normalized-angle 파생 계수의 마지막 bit와 exact metadata 비교가 충돌한 사건은 [공개 이력](https://github.com/Munsik-Kim/teacher-error-geometry-lab/blob/fd8bd773e2bfd0ec9da15fbe8d1447b8b0dff343/docs/PUBLICATION_HISTORY.md)에 남는다. 원 raw와 저장 cell coefficient의 exact 일치는 유지하고 수학적 재계산에만 기존 atol1e−10/rtol1e−9를 사용했다. 미세 오차에 맞춰 tolerance/seed/fitted risk를 고친 것이 아니다.

F0는 새 venv와 release 사본에서 저장248,000행·36 tests·선택 key 재생성·summary/그림을 다시 검산했다. 이번 closeout에서는 이 receipt와 입력 hash를 재사용하며 그 전체 검사를 반복하지 않는다. F0가 원문 §§3–4와 Table 1에 직접 접근한 receipt도 재사용한다. 이번 문서 작성자가 논문을 전면 새로 심사했다고 쓰지 않는다.

## Transformer 부록: 실제 실행과 멈춘 지점

모델 출처는 [HuggingFaceTB/SmolLM2-135M의 고정 revision](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2)이다. Revision은 `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, 실제 parameter 수는 **134,515,008**이다. 원 F2R identity receipt와 [facts의 F2R.identity](facts.json), [feature receipt metadata](inputs/metadata.json)를 대조했다. 저장 weight는 BF16, 실제 load/forward/cache는 **FP32**, backbone update는 **0**이었다. `model.safetensors`는 **269,060,552 bytes**, SHA256은 `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`이다. 이번 공개 편집에서 모델을 다시 다운로드·load·추론하지 않았으며, CPU asset에는 weight나 576D 전체 cache를 포함하지 않는다.

| 단계 | 실행·관측 | 보존한 판정 |
|---|---|---|
| F1 CPU sanity | 독립 Gaussian/비Gaussian mock·risk/sign identity·누수/rank 계약 | mock heldout norm drift CONTROL_INVALID. 실제 Transformer 결과 아님 |
| F2 PCA16 | SmolLM2-135M 고정 FP32 final-layer 마지막 non-padding token. cal2048/train0_2048/val512 = 4,608 unique 실제 feature. Gold val MSE .30068963811743066>.25 | COMPLETE_VALIDATION_STOP / WEAK_TASK_REPRESENTATION. Main112 actual0. H1/H2/H3 NOT_RUN |
| F2D | 동일 cache의 PCA16/64/full×OLS/ridge6개 gold readout, train-only CV, 오류 visibility 진단 | PCA_BOTTLENECK_SUPPORTED. 사후 개발 validation 결과이며 원 F2 실패를 PASS로 바꾸지 않음 |
| F2R technical v01 | 입력을 만들었지만 calibration replay의 4행/2048행 연결 오류를 strict shape check가 forward 이전 검출 | BLOCKED/GPU_WORKER_FAILED. forward0인 원 run·snapshot 보존 |
| F2R v02 | parent 입력 연결만 고치고 별도 run에서 같은 fresh4096 사례 재사용. Gold qualification .1830017484220633, projection 차이 .9318168707588288 | high teacher MSE .2333943610976521가 목표 .25의 ±5% 밖이므로 COMPLETE_QUALIFICATION_STOP. Main80 actual0/final0 |
| Closeout | 기존 F2R train0+qualification cache에 teacher-target4 OLS. 실제 크기당 전달률 G 및 독립 선형성 검산 | CLOSURE_PROTOCOL 이후 사후 fit4개 완료. Gold coefficient1개 재사용, 독립53 checks PASS. 기존 H1 등록결과·구간에는 합치지 않음 |

원 F2의 E gate는 pipeline 수준에서 통과했으나 CPU unit fixture가 공식 root2026102201의 test 첫128 scenario/input/gold를 mock 경로로 읽은 예외를 검출하지 못했다. **전역 blind test는 성립하지 않았다.** 실제 Transformer의 그 test feature/prediction/risk는 생성되지 않았다. 현재 fixture namespace2026102299 격리는 과거 노출을 취소하지 않는다. F2R은 알려진 기존4736 signature registry를 위해 정확히 노출됐던 prefix128만 재구성했으며 나머지1920개는 접근하지 않았다. 이번 closeout은 prefix128도 다시 생성하지 않고 접근 이력 문서만 참조한다.

F2의 원 실행14파일 snapshot과 이후 fixture/검산 도구 교정을 구분한다. 원 코드에 없던 resident hard watchdog 및 frozen prediction 동등성 대조 계약은 F2R에서 실제 CPU/CUDA 소유 worker 실패 주입·별도 process QR 대조로 보완했다. 이것은 모든 driver/kernel hang 복구의 보장이 아니다. F2R의 초기 recovery registry 참조 누락을 검출한 FAIL receipt와 byte-identical registry 포장 교정 뒤 PASS도 보존한다. 미실행 main 경로까지 검증 완료로 확대하지 않는다.

## 서로 다른 식별 객체

| 객체 | SHA 또는 SHA256 |
|---|---|
| P1 최종 실행 commit | df1173e91e77a43a53e9a82beea079c65bd0dab6 |
| P1 최종 source bundle | 286b7639b3abfb69503926dad32f35dc27ddd11356a8a15fe2b72377e37716a2 |
| P1 config | f909d298b234af6a3a1a4175066004bd203d91d0ac9912cb34271e058551a11b |
| P2 실행 commit | 9b502ee4739e7acb43f94a89bdf57cfbc303a38b |
| P2 source bundle | fc3f8e3e43beb117efd16c391d89a0d7b3d103611613d78639c869a01ddd20c7 |
| P2 provenance/config | 510781991629e037bd2dc1d66aa4533c7d903b3315b04050a53ee9cca2c1b4ef |
| v0.1.0 release | fd8bd773e2bfd0ec9da15fbe8d1447b8b0dff343 |
| 공개 evidence archive | fc01f26226732293d584c5d515452b665e1712b9c9759bc72c1717fff5b72d0a |
| F2 원 실행 source bundle | 776033209d8da800f7365376a50fffaf2d62095b77b24ab49d1f24040ba1b0a8 |
| F2 원 config / geometry | 69f123b293006dc2e67bf6dbade310ef1b04cafa92e54276f120d5f8e0e51970 / aa5c804beefc01c28f7f17f353c49c39c0723482e89b634d99c694d07451f946 |
| F2 사후 교정 도구 bundle | 69f8760f4d9a448b619c7c89a852afb2e77966b0bccf7e8b3704d3dde6d6bd22 |
| F2D 실행 source bundle | 648a3febaafce4c0bb3a1ac34ec493660d418b705bbfb741efdde89693569957 |
| F2R config | 4045f0cb6a54d99f305b51634cc6adbb492eeec1ca9c769b6a84ba9e297fc0f4 |
| F2R technical v01 / qualification v02 source | 56a87094179f4524d233f781c4e456751163291f354d196f5c15c9d9cfdd1409 / 15d2f30aa39e41cca54711642709735438c8cecc54b030d73ccd768a09ba4582 |
| Closeout 원 실행 source bundle | 8472854f0b38247df859e7ccd76d0ed684d407bdda251251e21c60c84576a799 |

Source bundle은 명시한 source 목록의 canonical bytes/hash 식별이고 Git commit과 다르다. 개별 config/geometry/input/result/archive hash도 서로 다른 객체다. 이번 source/config/input/result manifest는 새 closeout의 것만 관리하며 원 freeze나 evidence inventory를 갱신하지 않는다.

## Closeout 당시 로컬 통합 사본의 기준과 보존

`publication_snapshot/`은 읽기 전용 원 clone의 Git object에서 **0344833bbeaa96422ba5f3067b38f7e64e90516c**를 `git archive`로 export한 사본이다. 2026-10-22에 read-only `ls-remote`로 이 current main을 확인했고, [해당 CPU check](https://github.com/Munsik-Kim/teacher-error-geometry-lab/actions/runs/39704688110/job/119581831317)의 head SHA와 success도 확인했다. v0.1.0 이후의 원 차이는 README 두 개와 current publication/FILES.json뿐이며, old science/config/raw/evidence는 release와 byte-identical이다. 원 Git history/.git/venv/cache/model은 export하지 않았다.

부록과 README 링크는 그 **새 사본에만** 추가했다. `scripts/update_public_manifest.py`는 repository 전체 일반파일 목록도 검사하므로 새 부록을 current inventory에 포함했다. 공식 updater 실행·검사는 publication_snapshot 내부 current source_manifest/FILES에만 적용했다. 역사 evidence/manifest·records.tar.xz·recorded·원 과학 모듈은 변경하지 않았다. 정확한 export file hash와 사본 변경 목록은 원 closeout의 `audit/publication_export.json` 및 통합 receipt에 보존했다. 그 역사적 로컬 사본에는 새 commit/history나 원격 게시가 없었다. `publication_snapshot/` 디렉터리 전체를 이 부록에 중복 포함하지 않는다.

Closeout 당시 외부 접근은 공개 identity 확인을 위한 읽기 전용 `git ls-remote` 2회(첫 시도 DNS 실패, 두 번째 성공)와 GitHub check-runs GET 1회였다. 그 단계에서 새 CI를 trigger하거나 원격 내용을 변경하지 않았다. 모델 추론·generation·유료 API·클라우드 계산은0이었다. 이를 모든 종류의 네트워크/API 호출이0이었다고 표현하지 않는다.

## v0.2.0 공개 편집본과 배포용 inventory

이 부록의 공개 준비는 완료된 closeout 이후의 별도 문서·패키징 작업이다. 원본 closeout을 읽기 전용으로 두고, 현재 저장소의 `followups/linear_feature_closeout/`에 portable 입력·코드·저장 결과와 문서를 통합했다. 여기서 source/config/input/result/package manifest는 **현재 공개 파일의 범위**를 기술한다. 원 실행의 `results/closure_posthoc_01.source_freeze.json`과 `results/closure_posthoc_01.source_snapshot/`은 byte 그대로 보존한다. 실행 bundle, 새 공개 commit, current manifest와 release asset hash는 서로 다른 식별 객체다. 문서 정리와 archive 생성은 새 fit 또는 새로운 독립 평가가 아니다.

Portable checker의 로컬 패키징 교정 두 건도 과학 결과와 구분한다. 최초 도구는 README가 안내한 package 내부 `.venv`를 inventory에서 제외하지 않아 설치 직후 검사 실패를 일으켰다. 이를 제외한 v2에서는 `Path` 정렬과 manifest의 문자열 경로 정렬 차이가 드러났다. path→bytes/hash 사전은 같았고 순서만 달랐으며, 문자열 경로로 정렬하여 v3에서 실제 checker와 외부 재집계·독립 검산을 통과했다. 원 v1/v2 asset과 실패 기록은 로컬 closeout에 남아 있다. 네 primary OLS, config, frozen input 및 raw 결과는 이 교정 동안 변경하거나 재실행하지 않았다.

공개 문서를 편집한 뒤의 package inventory와 소규모 CPU asset은 새 current 값으로 생성한다. 역사 evidence archive·v0.1.0 tag·기존 science/config/raw·실행 snapshot은 갱신하지 않는다. 저장소 전체 공개 inventory와 이 부록의 scoped inventory를 각각 검사한다. 실제 원격 CI·commit·tag·release의 상태는 해당 게시 단계의 새 기록으로 확인하며, 과거 문서의 LOCAL_ONLY/NOT_RUN 상태를 소급 변경하지 않는다.

공개 준비의 이식성·metadata 변경도 실행 수치와 분리한다. 저장소 package version은 `pyproject.toml`과 `src/teacher_error_geometry/__init__.py`를 0.2.0으로 맞춘다. Wrapper의 cache/output 경로 정리는 새 clone에서 동작하도록 하는 이식성 수정이며, Gaussian generator·solver·risk나 closeout의 수학 함수·frozen input·저장 결과를 바꾸지 않는다. 원 실행 snapshot에는 당시 source가 그대로 남고, 새 공개 source manifest는 전달되는 현재 wrapper까지 식별한다.

## 검증의 의미와 남은 경계

독립 구현·수치 검산·저장기록 재현은 원 선형 이론을 실행하고 확인할 수 있는 산출물이다. 알려진 항등식 PASS는 새 정리의 발견이나 H1 SUPPORT가 아니다. G는 실제 eval error energy를 분모로 한 유한 표본 전이비이며 성공확률·정확도·population eta가 아니다. 한 train dataset의 사후 평가를8seed confidence interval로 포장하지 않는다.

AI가 구현·수식 대응·내부 수치 검산·그림 점검을 보조했다. 같은 머신의 새 process/venv와 원격 hosted CPU CI를 사람의 독립 심사로 표현하지 않는다. 자연 teacher 오류·생성형 KD·비선형 학생·학습되는 표현·MLP/LoRA·일반 압축 성능은 미검증이다. 이번 종료는 선형 이론의 독립 구현·부분 수치 재현·고정 feature 부록·재현 패키지의 완결이며, 새 연구를 자동 시작하지 않는다.
