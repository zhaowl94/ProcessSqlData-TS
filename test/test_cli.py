from __future__ import annotations

from io import StringIO
from pathlib import Path
import shutil
import unittest

from process_sql_data.cli import main
from test import temporary_directory


VALID_CONFIG = """\
[database]
host = "db.example.invalid"
port = 5432
database = "telemetry"
user = "readonly_user"
password = "test-password"
"""


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(temporary_directory()))
        self.repository = self.root / "ProcessSqlData-TS"
        self.repository.mkdir()
        self.entrypoint = self.repository / "ProcessSqlData Final.py"

    def test_validate_config_prints_no_secret(self) -> None:
        config = self.repository / "config.local.toml"
        config.write_text(VALID_CONFIG, encoding="utf-8")
        stdout = StringIO()
        stderr = StringIO()

        exit_code = main(
            ["validate-config", "--config", str(config)],
            entrypoint=self.entrypoint,
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(exit_code, 0)
        self.assertIn("forced read-only", stdout.getvalue())
        self.assertNotIn("test-password", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_invalid_config_returns_nonzero_without_connecting(self) -> None:
        missing = self.repository / "config.local.toml"
        stderr = StringIO()

        exit_code = main(
            ["validate-config", "--config", str(missing)],
            entrypoint=self.entrypoint,
            stdout=StringIO(),
            stderr=stderr,
        )

        self.assertEqual(exit_code, 2)
        self.assertIn("Configuration invalid", stderr.getvalue())

    def test_compare_returns_zero_only_for_compatible_trees(self) -> None:
        expected = self.root / "expected"
        actual = self.root / "actual"
        expected.mkdir()
        (expected / "result.csv").write_text(
            "id,value\n1,0.30000000000000004\n",
            encoding="utf-8",
        )
        shutil.copytree(expected, actual)
        stdout = StringIO()

        matching_exit = main(
            [
                "compare",
                "--expected",
                str(expected),
                "--actual",
                str(actual),
            ],
            entrypoint=self.entrypoint,
            stdout=stdout,
            stderr=StringIO(),
        )

        self.assertEqual(matching_exit, 0)
        self.assertIn("compatible", stdout.getvalue())

        (actual / "result.csv").write_text(
            "id,value\n2,0.3\n",
            encoding="utf-8",
        )
        stderr = StringIO()
        mismatching_exit = main(
            [
                "compare",
                "--expected",
                str(expected),
                "--actual",
                str(actual),
            ],
            entrypoint=self.entrypoint,
            stdout=StringIO(),
            stderr=stderr,
        )

        self.assertEqual(mismatching_exit, 1)
        self.assertIn("value differs", stderr.getvalue())

    def test_show_layout_lists_all_fourteen_paths_without_creating_them(self) -> None:
        stdout = StringIO()

        exit_code = main(
            ["show-layout"],
            entrypoint=self.entrypoint,
            stdout=stdout,
            stderr=StringIO(),
        )

        lines = stdout.getvalue().splitlines()
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(lines), 14)
        self.assertTrue(lines[0].startswith("DataRaw="))
        self.assertTrue(lines[-1].startswith("DataProcess13="))
        self.assertFalse((self.root / "DataRaw").exists())


if __name__ == "__main__":
    unittest.main()
