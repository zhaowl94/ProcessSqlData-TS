"""Persistent run manifests for explicit, validated resume behavior."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = 1
LEGACY_STAGE_NAMES = (
    "01_export_raw",
    "02_group_by_device",
    "03_parse_features",
    "04_merge_and_sort",
    "05_split_time_series",
    "06_remove_outliers",
    "07_plot_time_series",
    "08_classify_series",
    "09_convert_timestamps",
    "10_interpolate",
    "11_fit_arima",
    "12_plot_histograms",
    "13_summary_statistics",
    "14_linear_trend",
)


class RunMode(str, Enum):
    LEGACY = "legacy"
    CORRECTED = "corrected"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class InvalidStageTransition(ValueError):
    """Raised when stage state would make resume behavior ambiguous."""


@dataclass
class StageRecord:
    status: str = StageStatus.PENDING.value
    input_fingerprint: str | None = None
    output_fingerprint: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error_type: str | None = None


@dataclass
class RunManifest:
    run_id: str
    mode: str
    config_fingerprint: str
    input_fingerprint: str
    created_at: str
    stages: dict[str, StageRecord] = field(default_factory=dict)
    schema_version: int = MANIFEST_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        mode: RunMode,
        config_fingerprint: str,
        input_fingerprint: str,
    ) -> "RunManifest":
        return cls(
            run_id=run_id,
            mode=mode.value,
            config_fingerprint=_required_fingerprint(
                config_fingerprint,
                "config_fingerprint",
            ),
            input_fingerprint=_required_fingerprint(
                input_fingerprint,
                "input_fingerprint",
            ),
            created_at=_utc_now(),
            stages={name: StageRecord() for name in LEGACY_STAGE_NAMES},
        )

    def can_resume(
        self,
        *,
        mode: RunMode,
        config_fingerprint: str,
        input_fingerprint: str,
    ) -> bool:
        return (
            self.schema_version == MANIFEST_SCHEMA_VERSION
            and self.mode == mode.value
            and self.config_fingerprint == config_fingerprint
            and self.input_fingerprint == input_fingerprint
        )

    def mark_started(self, stage_name: str, *, input_fingerprint: str) -> None:
        stage = self._stage(stage_name)
        if stage.status not in {
            StageStatus.PENDING.value,
            StageStatus.FAILED.value,
        }:
            raise InvalidStageTransition(
                f"{stage_name} cannot start from state {stage.status!r}."
            )
        stage.status = StageStatus.RUNNING.value
        stage.input_fingerprint = _required_fingerprint(
            input_fingerprint,
            "input_fingerprint",
        )
        stage.output_fingerprint = None
        stage.started_at = _utc_now()
        stage.completed_at = None
        stage.error_type = None

    def mark_completed(
        self,
        stage_name: str,
        *,
        output_fingerprint: str,
    ) -> None:
        stage = self._stage(stage_name)
        if stage.status != StageStatus.RUNNING.value:
            raise InvalidStageTransition(
                f"{stage_name} cannot complete from state {stage.status!r}."
            )
        stage.status = StageStatus.COMPLETED.value
        stage.output_fingerprint = _required_fingerprint(
            output_fingerprint,
            "output_fingerprint",
        )
        stage.completed_at = _utc_now()

    def mark_failed(self, stage_name: str, *, error_type: str) -> None:
        stage = self._stage(stage_name)
        if stage.status != StageStatus.RUNNING.value:
            raise InvalidStageTransition(
                f"{stage_name} cannot fail from state {stage.status!r}."
            )
        stage.status = StageStatus.FAILED.value
        stage.error_type = _safe_error_type(error_type)
        stage.completed_at = _utc_now()

    def stage_can_be_reused(
        self,
        stage_name: str,
        *,
        input_fingerprint: str,
    ) -> bool:
        stage = self._stage(stage_name)
        return (
            stage.status == StageStatus.COMPLETED.value
            and stage.input_fingerprint == input_fingerprint
            and bool(stage.output_fingerprint)
        )

    def save(self, path: Path | str) -> None:
        """Atomically persist state without recording configuration values."""

        manifest_path = Path(path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = manifest_path.with_suffix(
            manifest_path.suffix + ".tmp"
        )
        payload = json.dumps(
            _manifest_to_dict(self),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        temporary_path.write_text(payload + "\n", encoding="utf-8")
        temporary_path.replace(manifest_path)

    @classmethod
    def load(cls, path: Path | str) -> "RunManifest":
        manifest_path = Path(path)
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load run manifest {manifest_path}: {exc}") from exc
        return _manifest_from_dict(raw)

    def _stage(self, stage_name: str) -> StageRecord:
        try:
            return self.stages[stage_name]
        except KeyError as exc:
            raise KeyError(f"Unknown pipeline stage: {stage_name}") from exc


def _manifest_to_dict(manifest: RunManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "run_id": manifest.run_id,
        "mode": manifest.mode,
        "config_fingerprint": manifest.config_fingerprint,
        "input_fingerprint": manifest.input_fingerprint,
        "created_at": manifest.created_at,
        "stages": {
            name: asdict(record) for name, record in manifest.stages.items()
        },
    }


def _manifest_from_dict(raw: Any) -> RunManifest:
    if not isinstance(raw, dict):
        raise ValueError("Run manifest root must be an object.")
    if raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported run manifest schema version.")

    raw_stages = raw.get("stages")
    if not isinstance(raw_stages, dict) or set(raw_stages) != set(
        LEGACY_STAGE_NAMES
    ):
        raise ValueError("Run manifest has an invalid stage set.")

    stages: dict[str, StageRecord] = {}
    allowed_stage_keys = set(StageRecord.__dataclass_fields__)
    for name in LEGACY_STAGE_NAMES:
        record = raw_stages[name]
        if not isinstance(record, dict) or set(record) != allowed_stage_keys:
            raise ValueError(f"Invalid stage record: {name}")
        stages[name] = StageRecord(**record)

    mode = raw.get("mode")
    if mode not in {item.value for item in RunMode}:
        raise ValueError(f"Invalid run mode: {mode!r}")

    return RunManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        run_id=str(raw.get("run_id", "")),
        mode=mode,
        config_fingerprint=_required_fingerprint(
            raw.get("config_fingerprint"),
            "config_fingerprint",
        ),
        input_fingerprint=_required_fingerprint(
            raw.get("input_fingerprint"),
            "input_fingerprint",
        ),
        created_at=str(raw.get("created_at", "")),
        stages=stages,
    )


def _required_fingerprint(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) < 16:
        raise ValueError(f"{name} must be an opaque fingerprint.")
    return value


def _safe_error_type(value: str) -> str:
    safe = "".join(character for character in value if character.isalnum() or character in "._")
    return safe[:128] or "UnknownError"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
