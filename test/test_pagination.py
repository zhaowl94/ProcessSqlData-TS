from __future__ import annotations

import unittest

from process_sql_data.pagination import (
    CorrectedKeysetPaginator,
    LegacyOffsetPaginator,
    PaginationSafetyError,
    primary_key_discovery_plan,
)


class PaginationTests(unittest.TestCase):
    def test_legacy_mode_explicitly_preserves_first_ten_thousand_skip(self) -> None:
        paginator = LegacyOffsetPaginator(
            schema="public",
            table="dashboard_dtudata",
        )

        first = paginator.next_page()
        second = paginator.next_page()

        self.assertNotIn("ORDER BY", first.query)
        self.assertEqual(first.parameters, (10_000, 10_000))
        self.assertEqual(second.parameters, (10_000, 20_000))

    def test_corrected_mode_starts_at_first_key_and_has_stable_order(self) -> None:
        paginator = CorrectedKeysetPaginator(
            schema="public",
            table="dashboard_dtudata",
            key_columns=("id",),
        )

        first = paginator.first_page()
        second = paginator.page_after((123,))

        self.assertEqual(
            first.query,
            'SELECT * FROM "public"."dashboard_dtudata" '
            'ORDER BY "id" LIMIT %s',
        )
        self.assertEqual(first.parameters, (10_000,))
        self.assertEqual(
            second.query,
            'SELECT * FROM "public"."dashboard_dtudata" '
            'WHERE ("id") > (%s) ORDER BY "id" LIMIT %s',
        )
        self.assertEqual(second.parameters, (123, 10_000))

    def test_corrected_mode_supports_composite_primary_key(self) -> None:
        paginator = CorrectedKeysetPaginator(
            schema="telemetry",
            table="measurements",
            key_columns=("device_id", "created_at"),
            batch_size=500,
        )

        page = paginator.page_after(("device-a", "2020-01-01"))

        self.assertIn(
            'WHERE ("device_id", "created_at") > (%s, %s)',
            page.query,
        )
        self.assertEqual(page.parameters, ("device-a", "2020-01-01", 500))

    def test_corrected_mode_refuses_missing_or_mismatched_key(self) -> None:
        with self.assertRaises(PaginationSafetyError):
            CorrectedKeysetPaginator(
                schema="public",
                table="dashboard_dtudata",
                key_columns=(),
            )

        paginator = CorrectedKeysetPaginator(
            schema="public",
            table="dashboard_dtudata",
            key_columns=("id", "created_at"),
        )
        with self.assertRaises(PaginationSafetyError):
            paginator.page_after((1,))

    def test_identifiers_cannot_inject_sql(self) -> None:
        with self.assertRaises(PaginationSafetyError):
            LegacyOffsetPaginator(
                schema="public",
                table='dashboard_dtudata; DROP TABLE users',
            )

    def test_primary_key_discovery_uses_parameters(self) -> None:
        plan = primary_key_discovery_plan(
            schema="public",
            table="dashboard_dtudata",
        )

        self.assertEqual(plan.parameters, ("public", "dashboard_dtudata"))
        self.assertIn("index_definition.indisprimary", plan.query)
        self.assertNotIn("dashboard_dtudata", plan.query)


if __name__ == "__main__":
    unittest.main()
