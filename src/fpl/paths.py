from __future__ import annotations

from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """Walk up from *start* (or cwd) until pyproject.toml is found."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here


def resolve_under_root(path: str | Path, *, root: Path | None = None) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (root or project_root()) / resolved
