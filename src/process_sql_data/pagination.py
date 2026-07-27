"""Explicit legacy and corrected PostgreSQL pagination plans."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence


LEGACY_BATCH_SIZE = 10_000
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PaginationSafetyError(ValueError):
    """Raised when deterministic corrected pagination cannot be guaranteed."""


@dataclass(frozen=True)
class PagePlan:
    query: str
    parameters: tuple[Any, ...]


class LegacyOffsetPaginator:
    """Reproduce the historical first-page skip for compatibility mode."""

    def __init__(
        self,
        *,
        schema: str,
        table: str,
        batch_size: int = LEGACY_BATCH_SIZE,
    ) -> None:
        self.qualified_table = _qualified_table(schema, table)
        self.batch_size = _positive_batch_size(batch_size)
        self._next_offset = self.batch_size

    def next_page(self) -> PagePlan:
        plan = PagePlan(
            query=f"SELECT * FROM {self.qualified_table} LIMIT %s OFFSET %s",
            parameters=(self.batch_size, self._next_offset),
        )
        self._next_offset += self.batch_size
        return plan


class CorrectedKeysetPaginator:
    """Build deterministic keyset pages ordered by a PostgreSQL primary key."""

    def __init__(
        self,
        *,
        schema: str,
        table: str,
        key_columns: Sequence[str],
        batch_size: int = LEGACY_BATCH_SIZE,
    ) -> None:
        if not key_columns:
            raise PaginationSafetyError(
                "Corrected pagination requires at least one stable key column."
            )
        self.qualified_table = _qualified_table(schema, table)
        self.key_columns = tuple(_quote_identifier(item) for item in key_columns)
        self.batch_size = _positive_batch_size(batch_size)

    def first_page(self) -> PagePlan:
        order_by = ", ".join(self.key_columns)
        return PagePlan(
            query=(
                f"SELECT * FROM {self.qualified_table} "
                f"ORDER BY {order_by} LIMIT %s"
            ),
            parameters=(self.batch_size,),
        )

    def page_after(self, last_key: Sequence[Any]) -> PagePlan:
        if len(last_key) != len(self.key_columns):
            raise PaginationSafetyError(
                "Last key width does not match corrected pagination key."
            )
        columns = ", ".join(self.key_columns)
        placeholders = ", ".join(["%s"] * len(self.key_columns))
        order_by = ", ".join(self.key_columns)
        return PagePlan(
            query=(
                f"SELECT * FROM {self.qualified_table} "
                f"WHERE ({columns}) > ({placeholders}) "
                f"ORDER BY {order_by} LIMIT %s"
            ),
            parameters=(*last_key, self.batch_size),
        )


def primary_key_discovery_plan(*, schema: str, table: str) -> PagePlan:
    """Return a parameterized query that discovers primary key column order."""

    return PagePlan(
        query=(
            "SELECT attribute.attname "
            "FROM pg_index AS index_definition "
            "JOIN pg_class AS table_definition "
            "ON table_definition.oid = index_definition.indrelid "
            "JOIN pg_namespace AS namespace "
            "ON namespace.oid = table_definition.relnamespace "
            "JOIN unnest(index_definition.indkey) WITH ORDINALITY "
            "AS key_column(attribute_number, key_order) ON true "
            "JOIN pg_attribute AS attribute "
            "ON attribute.attrelid = table_definition.oid "
            "AND attribute.attnum = key_column.attribute_number "
            "WHERE index_definition.indisprimary "
            "AND namespace.nspname = %s "
            "AND table_definition.relname = %s "
            "ORDER BY key_column.key_order"
        ),
        parameters=(schema, table),
    )


def _qualified_table(schema: str, table: str) -> str:
    return f"{_quote_identifier(schema)}.{_quote_identifier(table)}"


def _quote_identifier(value: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise PaginationSafetyError(
            f"Unsafe PostgreSQL identifier: {value!r}"
        )
    return f'"{value}"'


def _positive_batch_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PaginationSafetyError("Batch size must be a positive integer.")
    return value
