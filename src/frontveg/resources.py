"""Location of the external repositories and model checkpoints.

Resolution order for each resource:
1. its dedicated environment variable (e.g. ``FRONTVEG_SAM3_CKPT``);
2. ``<root>/external`` or ``<root>/checkpoints``, where ``<root>`` is ``FRONTVEG_ROOT``
   or the first FrontVeg V2 folder found by :func:`find_project_root`.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_ROOT = "FRONTVEG_ROOT"
ENV_DEPTH_REPO = "FRONTVEG_DEPTH_REPO"
ENV_DEPTH_CKPT_DIR = "FRONTVEG_DEPTH_CKPT_DIR"
ENV_SAM3_REPO = "FRONTVEG_SAM3_REPO"
ENV_SAM3_CKPT = "FRONTVEG_SAM3_CKPT"

# Folders searched (with their direct children) when nothing else matches.
_HOME_FOLDERS = ("", "Desktop", "Documents", "Downloads", "data", "projects", "repos", "workspace")


def _is_project_root(path: Path) -> bool:
    return (path / "external").is_dir() and (path / "checkpoints").is_dir()


def _children(path: Path) -> Iterator[Path]:
    try:
        yield from (child for child in path.iterdir() if child.is_dir())
    except OSError:
        return


def _candidates() -> Iterator[Path]:
    # Editable install: the package lives inside the repository.
    yield from Path(__file__).resolve().parents
    # Launched from inside or next to the repository.
    for folder in (Path.cwd(), *Path.cwd().parents):
        yield folder
        yield from _children(folder)
    # Repository cloned in a usual location of the home folder.
    home = Path.home()
    for name in _HOME_FOLDERS:
        folder = home / name
        if folder.is_dir():
            yield folder
            yield from _children(folder)


def find_project_root() -> Path | None:
    """Return the folder holding ``external/`` and ``checkpoints/``, or None."""
    if ENV_ROOT in os.environ:
        return Path(os.environ[ENV_ROOT]).expanduser().resolve()
    for folder in _candidates():
        if _is_project_root(folder):
            logger.info("FrontVeg V2 project root: %s", folder)
            return folder
    return None


@dataclass(frozen=True)
class ResourcePaths:
    """Paths to the external code and the checkpoints."""

    depth_repo: Path
    depth_ckpt_dir: Path
    sam3_repo: Path
    sam3_ckpt: Path

    def depth_ckpt(self, encoder: str) -> Path:
        return self.depth_ckpt_dir / f"depth_anything_v2_{encoder}.pth"

    def missing(self, encoder: str) -> list[str]:
        """Human-readable list of the resources that cannot be found."""
        checks = {
            "Depth-Anything-V2 repository": self.depth_repo,
            "Depth-Anything-V2 checkpoint": self.depth_ckpt(encoder),
            "SAM3 repository": self.sam3_repo,
            "SAM3 checkpoint": self.sam3_ckpt,
        }
        return [f"{name}: {path}" for name, path in checks.items() if not path.exists()]


def resolve_paths(root: Path | None = None) -> ResourcePaths:
    """Resolve every resource path, honouring the environment variables."""
    if root is None:
        root = find_project_root() or Path.cwd()

    def pick(env: str, default: Path) -> Path:
        return Path(os.environ[env]).expanduser() if env in os.environ else default

    return ResourcePaths(
        depth_repo=pick(ENV_DEPTH_REPO, root / "external" / "Depth-Anything-V2"),
        depth_ckpt_dir=pick(ENV_DEPTH_CKPT_DIR, root / "checkpoints" / "depthanything_ckpts"),
        sam3_repo=pick(ENV_SAM3_REPO, root / "external" / "sam3"),
        sam3_ckpt=pick(ENV_SAM3_CKPT, root / "checkpoints" / "sam3_ckpts" / "sam3.pt"),
    )


def add_to_sys_path(repo_path: str | Path | None) -> None:
    """Make a cloned (non-installed) repository importable."""
    if repo_path is None:
        return
    repo = str(Path(repo_path).resolve())
    if Path(repo).is_dir() and repo not in sys.path:
        sys.path.append(repo)
