from __future__ import annotations

import os
from pathlib import Path
import tomllib
import unittest
from unittest.mock import patch

from process_sql_data.config import ConfigError, load_config
from test import REPOSITORY_ROOT, temporary_directory


VALID_CONFIG = """\
[database]
host = "db.example.invalid"
port = 5432
database = "telemetry"
user = "readonly_user"
password = "local-test-password"
schema = "public"
table = "dashboard_dtudata"
connect_timeout_seconds = 10
statement_timeout_seconds = 120
"""


class ConfigTests(unittest.TestCase):
    def test_valid_local_config_loads_and_is_read_only(self) -> None:
        with temporary_directory() as directory:
            path = Path(directory) / "config.local.toml"
            path.write_text(VALID_CONFIG, encoding="utf-8")

            config = load_config(path)

        self.assertTrue(config.database.read_only)
        self.assertEqual(config.database.port, 5432)
        self.assertEqual(config.database.table, "dashboard_dtudata")
        self.assertNotIn("password", config.database.log_fields())
        self.assertNotIn("local-test-password", repr(config.database))

    def test_missing_config_fails_closed(self) -> None:
        with temporary_directory() as directory:
            path = Path(directory) / "config.local.toml"
            with self.assertRaisesRegex(ConfigError, "does not exist"):
                load_config(path)

    def test_template_placeholder_is_rejected(self) -> None:
        with temporary_directory() as directory:
            path = Path(directory) / "config.local.toml"
            path.write_text(
                VALID_CONFIG.replace(
                    'password = "local-test-password"',
                    'password = "replace-me"',
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "must be replaced"):
                load_config(path)

    def test_unknown_database_key_is_rejected(self) -> None:
        with temporary_directory() as directory:
            path = Path(directory) / "config.local.toml"
            path.write_text(
                VALID_CONFIG + 'read_only = false\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "Unknown key"):
                load_config(path)

    def test_environment_variable_does_not_override_local_file(self) -> None:
        with temporary_directory() as directory:
            path = Path(directory) / "config.local.toml"
            path.write_text(VALID_CONFIG, encoding="utf-8")
            with patch.dict(
                os.environ,
                {"PSDTS_DB_PASSWORD": "environment-password"},
            ):
                config = load_config(path)

        self.assertEqual(config.database.password, "local-test-password")

    def test_example_has_expected_shape_and_local_file_is_ignored(self) -> None:
        example = REPOSITORY_ROOT / "config.example.toml"
        with example.open("rb") as handle:
            parsed = tomllib.load(handle)

        self.assertEqual(
            set(parsed["database"]),
            {
                "host",
                "port",
                "database",
                "user",
                "password",
                "schema",
                "table",
                "connect_timeout_seconds",
                "statement_timeout_seconds",
            },
        )
        gitignore = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("config.local.toml", gitignore.splitlines())


if __name__ == "__main__":
    unittest.main()
