"""Safe Python 3 foundations for the ProcessSqlData-TS migration."""

from .comparison import ComparisonResult, Difference, compare_output_trees
from .config import AppConfig, ConfigError, DatabaseConfig, load_config
from .database import (
    DatabaseDependencyError,
    DatabaseSafetyError,
    connect_read_only,
)
from .exporter import ExportError, ExportSummary, export_raw_pages
from .layout import (
    LEGACY_OUTPUT_DIRECTORY_NAMES,
    PathSafetyError,
    ProjectLayout,
)
from .pagination import (
    CorrectedKeysetPaginator,
    LegacyOffsetPaginator,
    PagePlan,
    PaginationSafetyError,
)
from .runtime import (
    PublishError,
    PublishOutcome,
    prepare_staging,
    publish_staged_outputs,
    rollback_published_outputs,
)
from .run_state import (
    LEGACY_STAGE_NAMES,
    InvalidStageTransition,
    RunManifest,
    RunMode,
    StageStatus,
)

__all__ = [
    "AppConfig",
    "ComparisonResult",
    "ConfigError",
    "DatabaseDependencyError",
    "DatabaseConfig",
    "DatabaseSafetyError",
    "Difference",
    "ExportError",
    "ExportSummary",
    "LEGACY_OUTPUT_DIRECTORY_NAMES",
    "LEGACY_STAGE_NAMES",
    "LegacyOffsetPaginator",
    "InvalidStageTransition",
    "PathSafetyError",
    "PagePlan",
    "PaginationSafetyError",
    "ProjectLayout",
    "PublishError",
    "PublishOutcome",
    "CorrectedKeysetPaginator",
    "RunManifest",
    "RunMode",
    "StageStatus",
    "compare_output_trees",
    "connect_read_only",
    "export_raw_pages",
    "load_config",
    "prepare_staging",
    "publish_staged_outputs",
    "rollback_published_outputs",
]

__version__ = "0.1.0a0"
