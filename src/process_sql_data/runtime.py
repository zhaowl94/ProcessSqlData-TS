"""Staging, publish, and rollback without recursive deletion of live outputs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .layout import LEGACY_OUTPUT_DIRECTORY_NAMES, ProjectLayout


class PublishError(RuntimeError):
    """Raised when staged outputs cannot be published safely."""


@dataclass(frozen=True)
class PublishOutcome:
    """Paths affected by a completed publish or rollback operation."""

    run_id: str
    published: tuple[str, ...]
    backed_up: tuple[str, ...]


MoveFunction = Callable[[Path, Path], None]


def _move_path(source: Path, destination: Path) -> None:
    source.replace(destination)


def prepare_staging(
    layout: ProjectLayout,
    run_id: str,
    *,
    directory_names: Iterable[str] = LEGACY_OUTPUT_DIRECTORY_NAMES,
) -> tuple[Path, ...]:
    """Create empty staging directories for one isolated run."""

    prepared: list[Path] = []
    for directory_name in tuple(directory_names):
        staged = layout.staged_output_path(run_id, directory_name)
        if staged.exists():
            raise PublishError(f"Staging path already exists: {staged}")
        staged.mkdir(parents=True)
        prepared.append(staged)
    return tuple(prepared)


def publish_staged_outputs(
    layout: ProjectLayout,
    run_id: str,
    *,
    directory_names: Iterable[str] = LEGACY_OUTPUT_DIRECTORY_NAMES,
    _move: MoveFunction = _move_path,
) -> PublishOutcome:
    """Publish a complete staged tree and restore old outputs on any failure."""

    names = tuple(directory_names)
    staged_paths = {
        name: layout.staged_output_path(run_id, name) for name in names
    }
    target_paths = {name: layout.output_path(name) for name in names}
    backup_paths = {
        name: layout.backup_output_path(run_id, name) for name in names
    }

    for name in names:
        staged = staged_paths[name]
        backup = backup_paths[name]
        if not staged.is_dir():
            raise PublishError(f"Missing staged output directory: {staged}")
        if backup.exists():
            raise PublishError(f"Backup path already exists: {backup}")

    published: list[str] = []
    backed_up: list[str] = []
    moved_target_without_publish: str | None = None
    try:
        for name in names:
            target = target_paths[name]
            staged = staged_paths[name]
            backup = backup_paths[name]
            if target.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                _move(target, backup)
                backed_up.append(name)
                moved_target_without_publish = name
            _move(staged, target)
            published.append(name)
            moved_target_without_publish = None
    except Exception as exc:
        _restore_after_failed_publish(
            names=names,
            published=published,
            backed_up=backed_up,
            moved_target_without_publish=moved_target_without_publish,
            staged_paths=staged_paths,
            target_paths=target_paths,
            backup_paths=backup_paths,
            move=_move,
        )
        raise PublishError(f"Publish failed and was rolled back: {exc}") from exc

    return PublishOutcome(
        run_id=run_id,
        published=tuple(published),
        backed_up=tuple(backed_up),
    )


def rollback_published_outputs(
    layout: ProjectLayout,
    run_id: str,
    *,
    directory_names: Iterable[str] = LEGACY_OUTPUT_DIRECTORY_NAMES,
    _move: MoveFunction = _move_path,
) -> PublishOutcome:
    """Restore retained backups while preserving the replaced new outputs."""

    names = tuple(directory_names)
    run_root = layout.run_root(run_id)
    rolled_back_root = run_root / "rolled-back"
    restored: list[str] = []
    preserved_new: list[str] = []

    for name in names:
        target = layout.output_path(name)
        backup = layout.backup_output_path(run_id, name)
        preserved = rolled_back_root / name
        if not backup.is_dir():
            raise PublishError(f"Missing rollback backup: {backup}")
        if preserved.exists():
            raise PublishError(f"Rollback preservation path exists: {preserved}")
        if target.exists():
            preserved.parent.mkdir(parents=True, exist_ok=True)
            _move(target, preserved)
            preserved_new.append(name)
        _move(backup, target)
        restored.append(name)

    return PublishOutcome(
        run_id=run_id,
        published=tuple(restored),
        backed_up=tuple(preserved_new),
    )


def _restore_after_failed_publish(
    *,
    names: tuple[str, ...],
    published: list[str],
    backed_up: list[str],
    moved_target_without_publish: str | None,
    staged_paths: dict[str, Path],
    target_paths: dict[str, Path],
    backup_paths: dict[str, Path],
    move: MoveFunction,
) -> None:
    if moved_target_without_publish is not None:
        name = moved_target_without_publish
        backup = backup_paths[name]
        target = target_paths[name]
        if backup.exists() and not target.exists():
            move(backup, target)

    for name in reversed(published):
        target = target_paths[name]
        staged = staged_paths[name]
        backup = backup_paths[name]
        if target.exists() and not staged.exists():
            staged.parent.mkdir(parents=True, exist_ok=True)
            move(target, staged)
        if name in backed_up and backup.exists() and not target.exists():
            move(backup, target)
