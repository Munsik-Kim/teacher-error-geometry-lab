# Reproduce stored records

Use the installation and single `verify-public` command in [README](../README.md). Run from the public repository root using its installed project Python with `-I -B`. The driver discovers the checked-out files from cwd (or explicit `TEG_PROJECT_ROOT`). It unsets inherited PYTHONPATH/PYTHONHOME for children, disables user-site/bytecode, fixes child BLAS thread1, restores to a new external path and writes all new outputs outside the repository.

The integrated check runs 17 P1 + 12 P2 + 7 archive/tool tests (small fixtures, not full experiments), verifies all 168,000 P1 and 80,000 P2 canonical rows for keys/risk identities/pairing, and regenerates 12 preselected datasets in each phase. P1 verification makes 32 replay solver calls; P2 makes 60 lstsq calls (48 replay fits plus 12 mixed-target checks) and 48 QR validation solves. Unit-test calls are separate. It regenerates summaries and seven PNG/SVG pairs using untouched historical snapshot modules; numeric comparison retains atol=1e-10,rtol=1e-9. Figure byte equality is measured, not promised across platforms/renderers.

The first P1 full's original flags/source/raw are hash-checked and preserved. Its full sweep is not repeated. The same-sample second full is not counted as new evidence. Raw errors are never removed; saved summaries retain failed/missing denominators. No new statistical success rule is introduced.

## Restore only, then regenerate with installed public code

Each output must be new. These commands do not write into a recorded run:

```bash
.venv/bin/python -I -B -m teacher_error_geometry restore-recorded --output /tmp/teg-records-NEW_ID
.venv/bin/python -I -B -m teacher_error_geometry summarize --run /tmp/teg-records-NEW_ID/runs/p1_full_20261021_02 --output /tmp/teg-summary-p1-NEW_ID
.venv/bin/python -I -B -m teacher_error_geometry figures --summary /tmp/teg-summary-p1-NEW_ID --output /tmp/teg-figures-p1-NEW_ID
.venv/bin/python -I -B -m teacher_error_geometry p2 summarize --run /tmp/teg-records-NEW_ID/runs/p2_paper_anchor_20261021T065112Z --output /tmp/teg-summary-p2-NEW_ID
.venv/bin/python -I -B -m teacher_error_geometry p2 figures --summary /tmp/teg-summary-p2-NEW_ID --output /tmp/teg-figures-p2-NEW_ID
```

For full recorded integrity and seed checks use `verify-public`; directly invoking the current code's old `verify` on a historical run would compare different source versions and P1's default also writes a receipt inside that run. The integrated driver instead invokes the untouched frozen version with explicit external receipts and relocates P2's legacy preservation inventory. Keep restored evidence immutable.

## Optional future smoke/full

These optional commands create new outputs using the portable public source. They are not CI or stored-record verification steps. The two JSON configs remain the completed experiment settings. Fresh results have a new source freeze and must not be represented as the old recorded execution.

```bash
scripts/teg smoke --output /tmp/teg-p1-smoke-NEW
scripts/teg p2 smoke --output /tmp/teg-p2-smoke-NEW
scripts/teg run --config configs/p1.json --output /tmp/teg-p1-full-NEW
scripts/teg p2 run --config configs/p2_paper_anchor.json --output /tmp/teg-p2-full-NEW
```

A Python wheel contains code, not the 21MiB evidence; verification needs a checkout of the public files. Installation needs a package index or matching local wheels. After installation, recorded verification needs neither network, original research files nor an existing conda environment. The P3 test uses a new venv and new public copy on the same machine/OS, not another machine's independent execution.
