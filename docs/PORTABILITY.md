# Public packaging changes

P3 is separate from the original research worktree. The scientific generators, targets, OLS solvers, actual/expected risk functions, summary statistics, plots, recorded raw and original scorers are unchanged. A machine-readable original-to-public comparison is in [portability manifest](../publication/portability.json).

|Public change|Purpose|
|---|---|
|pyproject.toml, requirements-lock.txt|Normal package install; Python 3.11, NumPy 1.26.4, Matplotlib 3.10.9 and only their dependencies/build tools|
|paths.py; artifacts.py ROOT import|Find checked-out configs from cwd or explicit TEG_PROJECT_ROOT after a non-editable install|
|__main__.py|Add verify-public and restore-recorded; existing P1 commands keep their meanings|
|p2/cli.py|Legacy P1 inventory optional for new runs; omission is NOT_APPLICABLE, not invented preservation PASS|
|p2/provenance.py|Record null commit/UNVERSIONED_COPY when a clean source copy has no .git; numerical freeze/fit behavior unchanged|
|scripts/teg|Explicit TEG_PYTHON, project .venv or active python3; remove personal interpreter/PYTHONPATH; child BLAS thread1|
|public_archive.py, public_validation.py, recorded_worker.py, tests_public|Safe fresh-path evidence restoration, frozen-source replay, original-path denial and summary comparison|
|README/docs/data/publication/workflow|New bilingual explanations and local evidence; historical originals stay immutable|

The public worker's sole historical adaptation replaces P2 preservation's absolute path lookup with a relative-path lookup into restored records. It still checks the original 340-file sizes/digests. It does not patch risk, solver, distribution, RNG or statistical criteria. New receipts identify this adaptation. Old source bundles and configs are checked separately before execution.

The candidate's local commit/source inventory is a **new object**; it is not either historical execution commit or source bundle. The original Git history was not copied. A candidate inventory excludes itself and .git to avoid a self-referential hash. The archive hash authenticates only local bytes, not independent authorship or peer review.
