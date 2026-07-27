"""Repository-local launcher for safe migration utilities."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from process_sql_data.cli import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            entrypoint=REPOSITORY_ROOT / "ProcessSqlData Final.py",
        )
    )
