from __future__ import annotations

import codecs
from pathlib import Path
import shutil
import unittest

from process_sql_data.comparison import compare_output_trees
from test import temporary_directory


class ComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(self.enterContext(temporary_directory()))
        self.expected = root / "expected"
        self.actual = root / "actual"
        self._create_expected_tree()
        shutil.copytree(self.expected, self.actual)

    def _create_expected_tree(self) -> None:
        data_raw = self.expected / "DataRaw"
        summary = self.expected / "DataProcess12"
        empty_directory = self.expected / "DataProcess13" / "empty"
        data_raw.mkdir(parents=True)
        summary.mkdir(parents=True)
        empty_directory.mkdir(parents=True)
        (data_raw / "SqlData1.csv").write_text(
            "index,sn,value\n0,device-a,0.30000000000000004\n",
            encoding="utf-8",
        )
        (summary / "ResultAll.csv").write_text(
            "index,var,mean,file\n0,1.25,2.5,device-a_temp.csv\n",
            encoding="utf-8",
        )
        (summary / "result.txt").write_text(
            "synthetic result\n",
            encoding="utf-8",
        )

    def test_identical_trees_match(self) -> None:
        result = compare_output_trees(self.expected, self.actual)
        self.assertTrue(result.matches, result.summary())

    def test_float_formatting_within_tolerance_matches(self) -> None:
        path = self.actual / "DataRaw" / "SqlData1.csv"
        path.write_text(
            "index,sn,value\n0,device-a,0.3\n",
            encoding="utf-8",
        )
        result = compare_output_trees(self.expected, self.actual)
        self.assertTrue(result.matches, result.summary())

    def test_integer_representation_must_match_exactly(self) -> None:
        path = self.actual / "DataRaw" / "SqlData1.csv"
        path.write_text(
            "index,sn,value\n00,device-a,0.30000000000000004\n",
            encoding="utf-8",
        )
        result = compare_output_trees(self.expected, self.actual)
        self.assertFalse(result.matches)
        self.assertIn("value differs", result.summary())

    def test_header_and_row_order_differences_are_reported(self) -> None:
        path = self.actual / "DataProcess12" / "ResultAll.csv"
        path.write_text(
            "index,mean,var,file\n0,2.5,1.25,device-a_temp.csv\n"
            "1,3.0,2.0,device-b_temp.csv\n",
            encoding="utf-8",
        )
        result = compare_output_trees(self.expected, self.actual)
        self.assertFalse(result.matches)
        summary = result.summary()
        self.assertIn("header differs", summary)
        self.assertIn("row count differs", summary)

    def test_missing_and_unexpected_paths_are_reported(self) -> None:
        shutil.rmtree(self.actual / "DataProcess13")
        (self.actual / "Unexpected").mkdir()
        result = compare_output_trees(self.expected, self.actual)
        self.assertFalse(result.matches)
        summary = result.summary()
        self.assertIn("missing directory", summary)
        self.assertIn("unexpected directory", summary)

    def test_csv_encoding_difference_is_reported(self) -> None:
        path = self.actual / "DataRaw" / "SqlData1.csv"
        content = path.read_text(encoding="utf-8")
        path.write_bytes(codecs.BOM_UTF8 + content.encode("utf-8"))
        result = compare_output_trees(self.expected, self.actual)
        self.assertFalse(result.matches)
        self.assertIn("encoding differs", result.summary())

    def test_non_csv_content_difference_is_reported(self) -> None:
        path = self.actual / "DataProcess12" / "result.txt"
        path.write_text("different result\n", encoding="utf-8")
        result = compare_output_trees(self.expected, self.actual)
        self.assertFalse(result.matches)
        self.assertIn("file content differs", result.summary())


if __name__ == "__main__":
    unittest.main()
