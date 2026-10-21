# Execution and diagnostic history

P1 first full: p1_full_20261021_01. Its initial paired teacher-risk diagnostic omitted a structural equality for C,rho=0,theta=pi/4 versus 3pi/4. Tiny floating-point differences were divided by tiny empirical SE, producing the original investigation flags (about 17.97 and 42.25). This was a classification error in a mathematically deterministic comparison.

The equality classification was corrected **after observing those results**. A second full invocation, p1_full_20261021_02, used the same seed mapping and samples. Its raw manifests and per-cell values match the first. The diagnostic change is not described as unobserved preregistration. Both freezes, original judgments, sources, raw paths and dated history are preserved. [First judgment](../recorded/p1_first/provenance/verification.json), [first summary](../recorded/p1_first/summary/overview.json), [first freeze](../recorded/p1_first/provenance/freeze.json), [final summary](../recorded/p1/summary/overview.json).

P1's two full invocations total 336,000 fits but only 2,000 independent repeats per setting and 62,000 distinct dataset keys. The first-run raw is deduplicated losslessly in the archive; restoration creates both original relative paths with original bytes/hashes.

P2 p2_paper_anchor_20261021T065112Z adds only the fixed Table 1 Gaussian anchor. Its 20,000 datasets yield four paired fits each. The original preservation receipt covers 340 preexisting P1 files (92,111,045 bytes); it remains unchanged. P2's frozen scientific source matched the original research tree at P3 entry. The P1 historical snapshot matches its freeze; current shared CLI/docs had intentionally been extended during P2.

P3 creates a separate portable public candidate and a new clean-copy/venv check. It changes no original research files or measurements. Past execution commits identify the original repository; this public candidate starts a new local history and does not contain those old commits. Frozen source manifests provide the independent path to their bytes.

The original reports and dated histories are restored as docs/P1_REPORT.md, docs/P1_REPORT.ko.md, docs/P2_REPORT.md, docs/P2_REPORT.ko.md, docs/RUN_HISTORY.md and docs/P2_RUN_HISTORY.md. These are archived paths, not live file links; use the [restoration guide](EVIDENCE.md). Earlier NOT_RUN states are historical, not silently rewritten. [Current P3 status](P3_REPORT.md).
