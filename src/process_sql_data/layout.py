"""Path rules that preserve legacy outputs without trusting the current directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


LEGACY_OUTPUT_DIRECTORY_NAMES = (
    "DataRaw",
    "DataProcess1",
    "DataProcess2",
    "DataProcess3",
    "DataProcess4",
    "DataProcess5",
    "DataProcess6",
    "DataProcess7",
    "DataProcess8",
    "DataProcess9",
    "DataProcess10",
    "DataProcess11",
    "DataProcess12",
    "DataProcess13",
)
_OUTPUT_DIRECTORY_SET = frozenset(LEGACY_OUTPUT_DIRECTORY_NAMES)
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class PathSafetyError(ValueError):
    """Raised when a path could escape the agreed repository boundary."""


@dataclass(frozen=True)
class ProjectLayout:
    """Resolved repository, legacy data, staging, and backup paths."""

    repository_root: Path
    data_root: Path
    runtime_root: Path

    @classmethod
    def from_entrypoint(cls, entrypoint: Path | str) -> "ProjectLayout":
        """Anchor all paths to the entrypoint instead of the current directory."""

        repository_root = Path(entrypoint).resolve().parent
        return cls(
            repository_root=repository_root,
            data_root=repository_root.parent.resolve(),
            runtime_root=(repository_root / ".runtime").resolve(),
        )

    def output_path(self, directory_name: str) -> Path:
        """Return one allowed legacy output directory or reject the request."""

        if directory_name not in _OUTPUT_DIRECTORY_SET:
            raise PathSafetyError(
                f"Unknown legacy output directory: {directory_name!r}"
            )

        candidate = self.data_root / directory_name
        resolved_candidate = candidate.resolve()
        if resolved_candidate.parent != self.data_root:
            raise PathSafetyError(
                f"Output path escapes the legacy data root: {resolved_candidate}"
            )
        return candidate

    def run_root(self, run_id: str) -> Path:
        """Return a hidden per-run directory after validating its identifier."""

        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise PathSafetyError(f"Unsafe run identifier: {run_id!r}")
        return self.runtime_root / "runs" / run_id

    def staged_output_path(self, run_id: str, directory_name: str) -> Path:
        """Return the staging path corresponding to a legacy output directory."""

        self.output_path(directory_name)
        return self.run_root(run_id) / "outputs" / directory_name

    def backup_output_path(self, run_id: str, directory_name: str) -> Path:
        """Return the rollback path corresponding to a legacy output directory."""

        self.output_path(directory_name)
        return self.run_root(run_id) / "backup" / directory_name
