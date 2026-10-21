"""Project paths independent of a developer's interpreter or checkout."""
import os
from pathlib import Path


def project_root():
    explicit = os.environ.get("TEG_PROJECT_ROOT")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not (path / "configs/p1.json").is_file():
            raise ValueError("TEG_PROJECT_ROOT must point to this repository")
        return path
    for origin in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for path in (origin, *origin.parents):
            if (path / "pyproject.toml").is_file() and (path / "configs/p1.json").is_file():
                return path
    return Path.cwd().resolve()
