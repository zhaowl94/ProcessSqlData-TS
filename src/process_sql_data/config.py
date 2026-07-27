"""Strict loading for the local, untracked TOML configuration file."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import tomllib
from typing import Any, Mapping


DEFAULT_CONFIG_FILENAME = "config.local.toml"
_PLACEHOLDER_VALUES = {
    "",
    "change-me",
    "changeme",
    "replace-me",
    "replace_me",
    "todo",
}
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DATABASE_KEYS = {
    "host",
    "port",
    "database",
    "user",
    "password",
    "schema",
    "table",
    "connect_timeout_seconds",
    "statement_timeout_seconds",
}


class ConfigError(ValueError):
    """Raised when local configuration is absent, unsafe, or malformed."""


@dataclass(frozen=True)
class DatabaseConfig:
    """Validated read-only database connection settings."""

    host: str
    port: int
    database: str
    user: str
    password: str = field(repr=False)
    schema: str = "public"
    table: str = "dashboard_dtudata"
    connect_timeout_seconds: int = 15
    statement_timeout_seconds: int = 300

    @property
    def read_only(self) -> bool:
        """Read-only mode is mandatory and cannot be disabled by configuration."""

        return True

    def log_fields(self) -> dict[str, str | int | bool]:
        """Return connection metadata that is safe to write to logs."""

        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "schema": self.schema,
            "table": self.table,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "statement_timeout_seconds": self.statement_timeout_seconds,
            "read_only": True,
        }


@dataclass(frozen=True)
class AppConfig:
    """Application configuration loaded exclusively from a local TOML file."""

    database: DatabaseConfig
    source_path: Path


def default_config_path(repository_root: Path) -> Path:
    """Return the only default location for real local configuration."""

    return Path(repository_root).resolve() / DEFAULT_CONFIG_FILENAME


def load_config(path: Path | str) -> AppConfig:
    """Load and validate configuration without consulting environment variables."""

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigError(
            f"Local configuration file does not exist: {config_path}. "
            f"Copy config.example.toml to {DEFAULT_CONFIG_FILENAME}."
        )

    try:
        with config_path.open("rb") as handle:
            raw_config = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {config_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read {config_path}: {exc}") from exc

    _reject_unknown_keys(raw_config, {"database"}, "top level")
    database_section = raw_config.get("database")
    if not isinstance(database_section, Mapping):
        raise ConfigError("Configuration must contain a [database] table.")

    _reject_unknown_keys(database_section, _DATABASE_KEYS, "[database]")
    database = DatabaseConfig(
        host=_required_text(database_section, "host"),
        port=_bounded_int(database_section, "port", minimum=1, maximum=65535),
        database=_required_text(database_section, "database"),
        user=_required_text(database_section, "user"),
        password=_required_text(database_section, "password"),
        schema=_identifier(database_section.get("schema", "public"), "schema"),
        table=_identifier(
            database_section.get("table", "dashboard_dtudata"),
            "table",
        ),
        connect_timeout_seconds=_bounded_int(
            database_section,
            "connect_timeout_seconds",
            minimum=1,
            maximum=300,
            default=15,
        ),
        statement_timeout_seconds=_bounded_int(
            database_section,
            "statement_timeout_seconds",
            minimum=1,
            maximum=86_400,
            default=300,
        ),
    )
    return AppConfig(database=database, source_path=config_path)


def _reject_unknown_keys(
    section: Mapping[str, Any],
    allowed_keys: set[str],
    section_name: str,
) -> None:
    unknown = sorted(set(section) - allowed_keys)
    if unknown:
        joined = ", ".join(unknown)
        raise ConfigError(f"Unknown key(s) in {section_name}: {joined}")


def _required_text(section: Mapping[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str):
        raise ConfigError(f"[database].{key} must be a string.")

    stripped = value.strip()
    if stripped.lower() in _PLACEHOLDER_VALUES:
        raise ConfigError(f"[database].{key} must be replaced with a real value.")
    return stripped


def _identifier(value: Any, key: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ConfigError(
            f"[database].{key} must be a simple PostgreSQL identifier."
        )
    return value


def _bounded_int(
    section: Mapping[str, Any],
    key: str,
    *,
    minimum: int,
    maximum: int,
    default: int | None = None,
) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"[database].{key} must be an integer.")
    if not minimum <= value <= maximum:
        raise ConfigError(
            f"[database].{key} must be between {minimum} and {maximum}."
        )
    return value
