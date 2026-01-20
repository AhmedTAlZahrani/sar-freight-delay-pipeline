import pandas as pd
import numpy as np
from datetime import datetime


VALID_COMMODITIES = [
    "petrochemicals", "containers", "minerals",
    "cement", "food_supplies", "building_materials",
]

VALID_ORIGINS = [
    "Riyadh", "Dammam", "Jubail", "Sudair", "Ras Al Khair",
]

VALID_DESTINATIONS = VALID_ORIGINS

NULL_THRESHOLDS = {
    "shipment_id": 0.0,
    "origin": 0.0,
    "destination": 0.0,
    "commodity": 0.0,
    "weight_tons": 0.01,
    "container_count": 0.01,
    "scheduled_departure": 0.0,
    "actual_departure": 0.02,
    "scheduled_arrival": 0.0,
    "actual_arrival": 0.02,
    "delay_minutes": 0.01,
    "delay_cause": 0.01,
    "carrier_id": 0.01,
    "customer_id": 0.01,
}


class DataQualityChecker:
    """Run data quality validation checks on freight shipment data.

    Performs null checks, range validation, referential integrity,
    and duplicate detection. Generates a quality report and can
    halt the pipeline on critical failures.
    """

    def __init__(self):
        self.results = []
        self.critical_failures = []

    def run_all_checks(self, df):
        """Execute the full suite of data quality checks.

        Args:
            df: DataFrame to validate.

        Returns:
            Dict with overall pass/fail status and check details.
        """
        print("=" * 60)
        print("DATA QUALITY CHECKS")
        print("=" * 60)

        self._check_nulls(df)
        self._check_ranges(df)
        self._check_referential_integrity(df)
        self._check_duplicates(df)
        self._check_date_consistency(df)

        report = self._build_report()
        self._print_report(report)
        return report

    def _check_nulls(self, df):
        """Check null percentages against thresholds per column.

        Args:
            df: DataFrame to check.
        """
        print("\n[1/5] Null checks...")
        for col, threshold in NULL_THRESHOLDS.items():
            if col not in df.columns:
                self._record("null_check", col, "SKIP", f"Column not found")
                continue

            null_pct = df[col].isnull().mean()
            passed = null_pct <= threshold

            self._record(
                "null_check", col,
                "PASS" if passed else "FAIL",
                f"Null rate: {null_pct:.4%} (threshold: {threshold:.1%})",
                critical=not passed and threshold == 0.0,
            )

    def _check_ranges(self, df):
        """Validate numeric columns fall within expected ranges.

        Args:
            df: DataFrame to check.
        """
        print("[2/5] Range validation...")

        # Weight must be positive
        if "weight_tons" in df.columns:
            invalid = (df["weight_tons"] <= 0).sum()
            passed = invalid == 0
            self._record(
                "range_check", "weight_tons",
                "PASS" if passed else "FAIL",
                f"Non-positive values: {invalid:,}",
                critical=not passed,
            )

        # Delay must be non-negative
        if "delay_minutes" in df.columns:
            invalid = (df["delay_minutes"] < 0).sum()
            passed = invalid == 0
            self._record(
                "range_check", "delay_minutes",
                "PASS" if passed else "FAIL",
                f"Negative values: {invalid:,}",
                critical=not passed,
            )

        # Container count must be positive integer
        if "container_count" in df.columns:
            invalid = (df["container_count"] <= 0).sum()
            passed = invalid == 0
            self._record(
                "range_check", "container_count",
                "PASS" if passed else "FAIL",
                f"Non-positive values: {invalid:,}",
            )

    def _check_referential_integrity(self, df):
        """Validate categorical columns contain only valid values.

        Args:
            df: DataFrame to check.
        """
        print("[3/5] Referential integrity...")

        if "commodity" in df.columns:
            invalid = ~df["commodity"].isin(VALID_COMMODITIES)
            count = invalid.sum()
            self._record(
                "referential_integrity", "commodity",
                "PASS" if count == 0 else "FAIL",
                f"Invalid values: {count:,}",
                critical=count > 0,
            )

        if "origin" in df.columns:
            invalid = ~df["origin"].isin(VALID_ORIGINS)
            count = invalid.sum()
            self._record(
                "referential_integrity", "origin",
                "PASS" if count == 0 else "FAIL",
                f"Invalid origins: {count:,}",
                critical=count > 0,
            )

        if "destination" in df.columns:
            invalid = ~df["destination"].isin(VALID_DESTINATIONS)
            count = invalid.sum()
            self._record(
                "referential_integrity", "destination",
                "PASS" if count == 0 else "FAIL",
                f"Invalid destinations: {count:,}",
                critical=count > 0,
            )

        if "delay_cause" in df.columns:
            valid_causes = [
                "weather", "maintenance", "port_congestion",
                "customs_clearance", "loading_delay",
                "track_maintenance", "signal_failure", "none",
            ]
            invalid = ~df["delay_cause"].isin(valid_causes)
            count = invalid.sum()
            self._record(
                "referential_integrity", "delay_cause",
                "PASS" if count == 0 else "FAIL",
                f"Invalid causes: {count:,}",
            )

    def _check_duplicates(self, df):
        """Detect duplicate shipment IDs.

        Args:
            df: DataFrame to check.
        """
        print("[4/5] Duplicate detection...")

        if "shipment_id" in df.columns:
            dup_count = df["shipment_id"].duplicated().sum()
            passed = dup_count == 0
            self._record(
                "duplicate_check", "shipment_id",
                "PASS" if passed else "WARN",
                f"Duplicates found: {dup_count:,}",
            )

    def _check_date_consistency(self, df):
        """Validate date ordering: scheduled before actual, departure before arrival.

        Args:
            df: DataFrame to check.
        """
        print("[5/5] Date consistency...")

        datetime_cols = ["scheduled_departure", "scheduled_arrival",
                         "actual_departure", "actual_arrival"]
        has_dates = all(col in df.columns for col in datetime_cols)

        if not has_dates:
            self._record("date_check", "dates", "SKIP", "Date columns missing")
            return

        for col in datetime_cols:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df = df.copy()
                df[col] = pd.to_datetime(df[col])

        # Scheduled departure should be before scheduled arrival
        bad_schedule = (df["scheduled_departure"] > df["scheduled_arrival"]).sum()
        self._record(
            "date_check", "schedule_order",
            "PASS" if bad_schedule == 0 else "FAIL",
            f"Departure after arrival: {bad_schedule:,}",
            critical=bad_schedule > 0,
        )

        # Actual arrival should be on or after actual departure
        bad_actual = (df["actual_departure"] > df["actual_arrival"]).sum()
        self._record(
            "date_check", "actual_order",
            "PASS" if bad_actual == 0 else "WARN",
            f"Actual departure after arrival: {bad_actual:,}",
        )

    def _record(self, check_type, field, status, detail, critical=False):
        """Record a quality check result.

        Args:
            check_type: Category of check.
            field: Column or field checked.
            status: PASS, FAIL, WARN, or SKIP.
            detail: Description of finding.
            critical: Whether failure should halt the pipeline.
        """
        result = {
            "check_type": check_type,
            "field": field,
            "status": status,
            "detail": detail,
            "critical": critical,
            "timestamp": datetime.now().isoformat(),
        }
        self.results.append(result)

        if critical and status == "FAIL":
            self.critical_failures.append(result)

    def _build_report(self):
        """Compile quality check results into a report.

        Returns:
            Dict with pass rate, check counts, and details.
        """
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        warned = sum(1 for r in self.results if r["status"] == "WARN")

        return {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "warnings": warned,
            "pass_rate": round(passed / total * 100, 1) if total > 0 else 0,
            "critical_failures": len(self.critical_failures),
            "pipeline_halt": len(self.critical_failures) > 0,
            "checks": self.results,
        }

    def _print_report(self, report):
        """Print a formatted quality report.

        Args:
            report: Quality report dict.
        """
        print("\n" + "=" * 60)
        print("QUALITY REPORT")
        print("=" * 60)
        print(f"  Total checks: {report['total_checks']}")
        print(f"  Passed:       {report['passed']}")
        print(f"  Failed:       {report['failed']}")
        print(f"  Warnings:     {report['warnings']}")
        print(f"  Pass rate:    {report['pass_rate']}%")
        print(f"  Critical:     {report['critical_failures']}")

        if report["pipeline_halt"]:
            print("\n  ** PIPELINE HALT: Critical quality gate failures detected **")
            for failure in self.critical_failures:
                print(f"    - [{failure['field']}] {failure['detail']}")
        else:
            print("\n  Quality gate PASSED. Pipeline may continue.")

    def get_report_df(self):
        """Return quality results as a DataFrame.

        Returns:
            DataFrame with one row per check.
        """
        return pd.DataFrame(self.results)

    def should_halt(self):
        """Check whether the pipeline should halt due to critical failures.

        Returns:
            True if any critical quality check failed.
        """
        return len(self.critical_failures) > 0

