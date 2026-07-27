from __future__ import annotations

import unittest

from process_sql_data.config import DatabaseConfig
from process_sql_data.database import (
    DatabaseSafetyError,
    connect_read_only,
)


class FakeCursor:
    def __init__(self, read_only_state: str = "on") -> None:
        self.read_only_state = read_only_state
        self.executions: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    def execute(
        self,
        query: str,
        parameters: tuple[object, ...] = (),
    ) -> None:
        self.executions.append((query, parameters))

    def fetchone(self) -> tuple[str]:
        return (self.read_only_state,)

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self, read_only_state: str = "on") -> None:
        self.fake_cursor = FakeCursor(read_only_state)
        self.session_arguments: dict[str, bool] | None = None
        self.closed = False

    def set_session(self, *, readonly: bool, autocommit: bool) -> None:
        self.session_arguments = {
            "readonly": readonly,
            "autocommit": autocommit,
        }

    def cursor(self) -> FakeCursor:
        return self.fake_cursor

    def close(self) -> None:
        self.closed = True


def database_config() -> DatabaseConfig:
    return DatabaseConfig(
        host="db.example.invalid",
        port=5432,
        database="telemetry",
        user="readonly_user",
        password="test-password",
        statement_timeout_seconds=120,
    )


class DatabaseSafetyTests(unittest.TestCase):
    def test_connection_is_forced_read_only_and_timeout_is_set(self) -> None:
        connection = FakeConnection()
        connector_arguments: dict[str, object] = {}

        def connector(**kwargs: object) -> FakeConnection:
            connector_arguments.update(kwargs)
            return connection

        result = connect_read_only(database_config(), connector=connector)

        self.assertIs(result, connection)
        self.assertEqual(
            connection.session_arguments,
            {"readonly": True, "autocommit": False},
        )
        self.assertEqual(
            connection.fake_cursor.executions,
            [
                (
                    "SELECT set_config('statement_timeout', %s, false)",
                    ("120000ms",),
                ),
                ("SHOW transaction_read_only", ()),
            ],
        )
        self.assertTrue(connection.fake_cursor.closed)
        self.assertFalse(connection.closed)
        self.assertEqual(connector_arguments["application_name"], "ProcessSqlData-TS")
        self.assertEqual(connector_arguments["password"], "test-password")

    def test_unconfirmed_read_only_state_closes_connection(self) -> None:
        connection = FakeConnection(read_only_state="off")

        with self.assertRaises(DatabaseSafetyError):
            connect_read_only(
                database_config(),
                connector=lambda **_: connection,
            )

        self.assertTrue(connection.closed)
        self.assertTrue(connection.fake_cursor.closed)


if __name__ == "__main__":
    unittest.main()
