# First public publication / 첫 공개 이력

2026-10-21 KST. The user explicitly authorized repository creation, the initial main push, necessary follow-up PRs, normal merge, tag and release. Repository: [Munsik-Kim/teacher-error-geometry-lab](https://github.com/Munsik-Kim/teacher-error-geometry-lab), public, main. The original research repository and its history were not pushed.

The reviewed P3 candidate was first published unchanged at `97ab0f92b20971d7eb947bf21d6cf7acc6c2f2af`. The [first CPU run](https://github.com/Munsik-Kim/teacher-error-geometry-lab/actions/runs/39568629688) installed dependencies successfully but failed in the P1 recorded-data worker. Its internal receipt was written only on the ephemeral runner and was not exposed by that workflow. The exact cause of that initial failure is **not established**; no numerical cause is asserted or hidden.

[PR 1](https://github.com/Munsik-Kim/teacher-error-geometry-lab/pull/1) first adds always-visible child logs and verification receipts, plus canonical public-manifest maintenance. With no numerical or acceptance-rule change, its [diagnostic run](https://github.com/Munsik-Kim/teacher-error-geometry-lab/actions/runs/39569273205) passes 35 tests and all 248,000 canonical fit rows. Its nine summaries and fourteen PNG/SVG files match the original bytes. This later success does not establish the cause of the first failure.

The same PR corrects a metadata assumption in the public wrapper: a generic verifier must not assert that it runs on the original research machine. New receipts record the actual platform/CPU/libc and hosted-CI context; unknown machine identity is null. The original P3 local receipt remains byte-original and its same-machine statement describes that historical local execution. The recorded-data verifier itself does not perform remote publication. Hosted CI is a separate CPU execution environment, not independent human review.

CI is pinned to Ubuntu 24.04, the image actually used by the successful diagnostic run, with Python 3.11 and the existing CPU lock. This avoids implicitly expanding validation to the announced ubuntu-latest image migration. All archive/source/raw/receipt integrity checks, isolation checks, 35 tests, seed replay, risk identities, paired summaries and original tolerances remain required. PR jobs and main push jobs remain enabled; extra branch/tag push jobs are avoided.

Current public manifests are updated with `python -I -B scripts/update_public_manifest.py` and checked without writes using `--check`. They exclude their own inventory hash as before and are distinct from historical execution freezes. The original archive is not regenerated. README publication links are updated; P1/P2/P3 historical reports and frozen status/receipt files are not rewritten.

최초 실패 메일은 최초 main push 뒤의 CPU CI 실패 알림입니다. 파일 업로드는 완료됐습니다. 초기 실패의 내부 원인은 미확정으로 남기며, 진단 누락을 수정했습니다. 수치 계산·표본·tolerance·통계 문턱을 바꿔 통과시킨 것이 아닙니다. 최종 main의 정확한 SHA와 성공 CI, v0.1.0 대상은 GitHub 릴리스 기록과 외부 인계에서 확인합니다. 과거 NOT_RUN은 당시 기록입니다.

No new full sweep, GPU/model request, paper-wide reproduction, paid API, Pages or inference-lab change is part of this publication. A release is permitted only after the final main commit's CPU workflow succeeds; the release record provides that exact SHA/run link.

## Recorded coefficient metadata portability / 동결 계수 metadata 이식성

The first merged-main check at `a5fae71b9c5ff38379db48d6fe7d67f78ac96be3` failed again: [run 39570663413](https://github.com/Munsik-Kim/teacher-error-geometry-lab/actions/runs/39570663413). The new diagnostics exposed 8,000 failed P1 checks, with the listed issues being exact comparisons of recomputed normalized-angle delta1/delta2 to recorded raw metadata. All 168,000 raw rows were finite/ok and statistical comparison remained NO_4SE_FLAGS. This distinguishes the metadata portability bug from an upload failure or changed Monte Carlo result. The earlier initial run lacks the inner receipt, so its identical cause is an inference rather than a recovered fact.

The configured normalized-angle coefficients involve floating-point sin/cos, a quadratic form and normalization. CPU/BLAS arithmetic can alter their last bits. Comparing newly derived values by `==` to historical serialized coefficients is inappropriate when the actual run already saved its exact coefficients in provenance/cells.json.

The public worker now validates every recomputed cell against that hash-checked recorded plan. Literal fields and schema must agree exactly; derived delta values must agree under the **original** atol=1e-10,rtol=1e-9. The untouched historical verifier then receives the recorded plan, so its raw-to-cell coefficient comparisons remain **exact equality**. Existing source/raw hashes, keys, pairing, fixed seed replay, scalar/QR checks, risk identities, summaries, tolerances and statistical thresholds are unchanged. No fitted coefficient or measured risk is corrected. New receipts list the actual recomputed-versus-recorded differences.

A regression test injects one ULP into a recomputed coefficient and verifies that the recorded value remains exact; it also rejects a changed literal rho and an out-of-tolerance coefficient. This adds one meaningful test: the current suite has 36 tests (17 P1 + 12 P2 + 7 public/tool). The historical P3 and PR 1 suites remain 35, not retroactively 36.

원인은 정규화 각도의 파생 계수를 다른 CPU에서 다시 계산한 뒤 직렬화된 원 계수와 `==`로 비교하던 검증 경계였습니다. 원 raw·동결 소스·역사적 receipt는 고치지 않습니다. 원래 저장한 계수와 raw의 정확 일치는 유지하고, 수학적 재계산과의 대조에만 이미 동결된 FP64 tolerance를 사용합니다. 사후 tolerance 완화나 seed 변경으로 통과시키지 않습니다.
