"""Database connection helpers with mandatory read-only enforcement."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from .config import DatabaseConfig


class DatabaseDependencyError(RuntimeError):
    """Raised when the PostgreSQL driver has not been installed."""


class DatabaseSafetyError(RuntimeError):
    """Raised when a database session cannot prove it is read-only."""


class CursorProtocol(Protocol):
    def execute(self, query: str, parameters: tuple[Any, ...] = ()) -> Any: ...

    def fetchone(self) -> tuple[Any, ...] | None: ...

    def fetchall(self) -> list[tuple[Any, ...]]: ...

    def close(self) -> Any: ...


class ConnectionProtocol(Protocol):
    def set_session(self, *, readonly: bool, autocommit: bool) -> Any: ...

    def cursor(self) -> CursorProtocol: ...

    def close(self) -> Any: ...


Connector = Callable[..., ConnectionProtocol]


def connect_read_only(
    config: DatabaseConfig,
    *,
    connector: Connector | None = None,
) -> ConnectionProtocol:
    """Connect and verify a read-only PostgreSQL session.

    The connector is injectable so safety behavior can be tested without a
    database or installed PostgreSQL driver.
    """

    if connector is None:
        try:
            import psycopg2
        except ImportError as exc:
            raise DatabaseDependencyError(
                "PostgreSQL support is not installed. Install the locked "
                "project dependency before attempting a database connection."
            ) from exc
        connector = psycopg2.connect

    connection = connector(
        host=config.host,
        port=config.port,
        dbname=config.database,
        user=config.user,
        password=config.password,
        connect_timeout=config.connect_timeout_seconds,
        application_name="ProcessSqlData-TS",
    )
    try:
        enforce_read_only(
            connection,
            statement_timeout_seconds=config.statement_timeout_seconds,
        )
    except Exception:
        connection.close()
        raise
    return connection


def enforce_read_only(
    connection: ConnectionProtocol,
    *,
    statement_timeout_seconds: int,
) -> None:
    """Set and verify read-only transaction behavior for one connection."""

    connection.set_session(readonly=True, autocommit=False)
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT set_config('statement_timeout', %s, false)",
            (f"{statement_timeout_seconds * 1000}ms",),
        )
        cursor.execute("SHOW transaction_read_only")
        state = cursor.fetchone()
    finally:
        cursor.close()

    if not state or str(state[0]).strip().lower() not in {"on", "true", "1"}:
        raise DatabaseSafetyError(
            "PostgreSQL did not confirm a read-only transaction."
        )
