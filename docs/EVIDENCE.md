# Evidence layout and restoration

[Manifest](../evidence/manifest.json) maps every original relative path, byte size and SHA256 to its compressed member and optional ordinary public copy. [records.tar.xz](../evidence/records.tar.xz) is one lossless archive: 22,178,200 bytes, SHA256 `fc01f26226732293d584c5d515452b665e1712b9c9759bc72c1717fff5b72d0a`. This archive hash is different from either execution's source bundle hash and either frozen config hash.

The archive contains 530 logical files stored as 257 unique regular members. Duplicate content (including identical P1 full raw) is stored once; `stored_as` aliases restore with ordinary file copies to **all original relative paths**, followed by bytes/SHA checks. No tar symlinks, hardlinks, duplicate names, absolute/traversal paths, .git or environment/cache files are accepted. Extraction requires a fresh destination and checks the archive digest before opening members.

```bash
.venv/bin/python -I -B -m teacher_error_geometry restore-recorded --output /tmp/teg-records-NEW_ID
```

Restoration recreates runs/p1_full_20261021_01, runs/p1_full_20261021_02, runs/p2_paper_anchor_20261021T065112Z and the original docs/configs/source/history, preflight records, snapshots and receipts. Prior handoffs restore under handoffs/. The P2 external historical rebuild restores under external_reproductions/p2/. Existing original relative paths are retained; these two outside-root locations have explicit path mappings.

Small summaries, figures and selected receipts are also present as ordinary files in [recorded/p1](../recorded/p1/summary/overview.json), [recorded/p1_first](../recorded/p1_first/summary/overview.json), [recorded/p2](../recorded/p2/summary/overview.json). Their bytes match the corresponding original files. These live links do not pretend that compressed members are normal GitHub files. Historical reports are **not** edited into public prose; public explanations are new files and original reports remain in the archive.

Historical metadata and frozen sources retain their original machine paths to preserve hashes. They are labels, not prerequisites. The public worker imports only the restored frozen snapshot, denies access to original research/handoff paths and socket connections, and adapts only P2's old preservation-inventory path lookup. New verification receipts/summaries/figures live outside the restored evidence, whose complete hash inventory is checked again afterward.

Omissions: original .git history, virtual/conda environments, .cache, __pycache__, pyc and unrelated projects. No paper HTML/PDF or author code is bundled. Historical generated figures/rebuilds were kept where present; aliases reduce duplicates. The [manifest](../evidence/manifest.json) lists all included paths and these exclusions. Source/raw/receipt hashes are verification evidence, not cryptographic signatures from an external party.

[Reproduction commands](REPRODUCTION.md) · [Historical correction](HISTORY.md) · [Current packaging changes](PORTABILITY.md)
