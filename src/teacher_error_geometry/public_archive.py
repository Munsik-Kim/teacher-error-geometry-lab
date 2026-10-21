"""Hash-checked, regular-file-only restoration of immutable recorded evidence."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative_name(name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("Invalid archive path")
    p = PurePosixPath(name)
    if not p.parts or p.is_absolute() or any(s in ("", ".", "..", ".git") for s in p.parts) or ":" in p.parts[0] or p.as_posix() != name:
        raise ValueError("Archive paths must be canonical relative paths")
    return p


def load_manifest(project):
    project = Path(project)
    manifest = json.loads((project / "evidence/manifest.json").read_text())
    entries = {}
    for entry in manifest["entries"]:
        name = relative_name(entry["path"]).as_posix()
        relative_name(entry["stored_as"])
        if name in entries:
            raise ValueError("Duplicate logical evidence path")
        if not isinstance(entry["bytes"], int) or entry["bytes"] < 0:
            raise ValueError("Invalid recorded file size")
        if len(entry["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in entry["sha256"]):
            raise ValueError("Invalid recorded digest")
        entries[name] = entry
    for entry in entries.values():
        stored = entries.get(entry["stored_as"])
        if stored is None or stored["stored_as"] != stored["path"] or (stored["bytes"], stored["sha256"]) != (entry["bytes"], entry["sha256"]):
            raise ValueError("Alias must refer directly to a matching regular member")
    return manifest, entries


def check_tree(root, entries):
    root = Path(root)
    changed = []
    for name, entry in entries.items():
        path = root.joinpath(*relative_name(name).parts)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            changed.append(name)
    if changed:
        raise ValueError("Restored evidence mismatch: " + repr(changed[:10]))
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != set(entries):
        raise ValueError("Unexpected restored evidence paths: " + repr(sorted(actual - set(entries))[:10]))
    return {"files": len(entries), "bytes": sum(e["bytes"] for e in entries.values()), "status": "PASS"}


def restore(project, output):
    project, output = Path(project), Path(output)
    manifest, entries = load_manifest(project)
    archive = project / "evidence" / relative_name(manifest["archive"]["path"])
    if archive.stat().st_size != manifest["archive"]["bytes"] or digest(archive) != manifest["archive"]["sha256"]:
        raise ValueError("Compressed evidence size/hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    physical = {name: e for name, e in entries.items() if e["stored_as"] == name}
    seen = set()
    with tarfile.open(archive, "r:xz") as tf:
        for member in tf:
            name = relative_name(member.name).as_posix()
            if name in seen or name not in physical or not member.isreg():
                raise ValueError("Duplicate, unexpected, or non-regular archive member")
            entry = physical[name]
            if member.size != entry["bytes"]:
                raise ValueError("Archive member size mismatch")
            seen.add(name)
            target = output.joinpath(*relative_name(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            remaining = member.size
            with tf.extractfile(member) as source, target.open("xb") as sink:
                while remaining:
                    data = source.read(min(1024 * 1024, remaining))
                    if not data:
                        raise ValueError("Truncated archive member")
                    sink.write(data); h.update(data); remaining -= len(data)
            if h.hexdigest() != entry["sha256"]:
                raise ValueError("Archive member hash mismatch")
            target.chmod(member.mode & 0o777)
    if seen != set(physical):
        raise ValueError("Archive is missing expected members")
    for name, entry in entries.items():
        if entry["stored_as"] == name:
            continue
        target = output.joinpath(*relative_name(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        source = output.joinpath(*relative_name(entry["stored_as"]).parts)
        with source.open("rb") as a, target.open("xb") as b:
            shutil.copyfileobj(a, b)
        target.chmod(source.stat().st_mode & 0o777)
    result = check_tree(output, entries)
    result.update(archive_sha256=manifest["archive"]["sha256"], physical_members=len(physical))
    return result
