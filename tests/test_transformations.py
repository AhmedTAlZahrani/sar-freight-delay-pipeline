import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.transformations import DataTransformer, SLA_THRESHOLDS, HAJJ_MONTHS, RAMADAN_MONTHS, SUMMER_MONTHS


def _make_shipment_rows(n=10, base_date=None, commodity="containers", origin="Riyadh",
                        destination="Dammam", delay_minutes=90):
    """Build a minimal shipment DataFrame suitable for DataTransformer.

    Covers all columns the transformer expects so tests don't blow up
    on missing fields.
    """
    base_date = base_date or datetime(2023, 7, 15, 8, 0)
    rows = []
    for i in range(n):
        sched_dep = base_date + timedelta(hours=i)
        sched_arr = sched_dep + timedelta(hours=4)
        act_dep = sched_dep + timedelta(minutes=delay_minutes // 2)
        act_arr = sched_arr + timedelta(minutes=delay_minutes)
        rows.append({
            "shipment_id": f"TEST-{i:04d}",
            "origin": origin,
            "destination": destination,
            "commodity": commodity,
            "weight_tons": 50.0 + i,
            "container_count": 2,
            "scheduled_departure": sched_dep,
            "actual_departure": act_dep,
            "scheduled_arrival": sched_arr,
            "actual_arrival": act_arr,
            "delay_minutes": delay_minutes,
            "delay_cause": "port_congestion" if delay_minutes > 0 else "none",
            "carrier_id": "SAR-C001",
            "customer_id": "CUST-0001",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def transformer():
    return DataTransformer()


@pytest.fixture
def raw_df():
    """Standard 10-row shipment frame in July (Hajj + summer)."""
    return _make_shipment_rows(n=10)


@pytest.fixture
def on_time_df():
    return _make_shipment_rows(n=5, delay_minutes=0)


@pytest.fixture
def mixed_commodity_df():
    """DataFrame with multiple commodities and varying delays."""
    frames = []
    for commodity, delay in [("petrochemicals", 200), ("food_supplies", 80),
                              ("minerals", 0), ("cement", 300)]:
        frames.append(_make_shipment_rows(n=3, commodity=commodity, delay_minutes=delay))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def multi_day_df():
    # Span 60 days so rolling windows have something to aggregate
    rows = []
    for day_offset in range(60):
        base = datetime(2023, 3, 1) + timedelta(days=day_offset)
        sub = _make_shipment_rows(n=2, base_date=base, delay_minutes=day_offset * 5)
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def transformed_df(transformer, raw_df):
    """Convenience: raw_df already pushed through transform()."""
    return transformer.transform(raw_df)


# ---------------------------------------------------------------------------
# _ensure_datetime
# ---------------------------------------------------------------------------

class TestEnsureDatetime:
    def test_converts_string_columns(self, transformer):
        df = _make_shipment_rows(n=2)
        # force strings
        for col in ["scheduled_departure", "actual_departure",
                     "scheduled_arrival", "actual_arrival"]:
            df[col] = df[col].astype(str)

        result = transformer._ensure_datetime(df)
        assert pd.api.types.is_datetime64_any_dtype(result["scheduled_departure"])
        assert pd.api.types.is_datetime64_any_dtype(result["actual_arrival"])

    def test_leaves_datetime_untouched(self, transformer, raw_df):
        result = transformer._ensure_datetime(raw_df)
        assert result["scheduled_departure"].equals(raw_df["scheduled_departure"])


# ---------------------------------------------------------------------------
# _calculate_delay_hours
# ---------------------------------------------------------------------------

class TestDelayHoursCalculation:
    """Verify delay_hours and departure_delay_hours derivations."""

    def test_delay_hours_added(self, transformer, raw_df):
        out = transformer._calculate_delay_hours(raw_df.copy())
        assert "delay_hours" in out.columns
        assert "departure_delay_hours" in out.columns

    def test_delay_hours_value(self, transformer):
        df = _make_shipment_rows(n=1, delay_minutes=120)
        out = transformer._calculate_delay_hours(df)
        assert out["delay_hours"].iloc[0] == pytest.approx(2.0, abs=0.05)

    def test_zero_delay(self, transformer, on_time_df):
        out = transformer._calculate_delay_hours(on_time_df)
        assert (out["delay_hours"] == 0.0).all()


# ---------------------------------------------------------------------------
# _flag_sla_violations
# ---------------------------------------------------------------------------

class TestSlaViolations:
    def test_sla_columns_created(self, transformer, raw_df):
        out = transformer._flag_sla_violations(raw_df.copy())
        assert "sla_violated" in out.columns
        assert "sla_threshold_minutes" in out.columns

    def test_containers_below_threshold(self, transformer):
        # containers threshold is 180 min; delay=90 -> no violation
        df = _make_shipment_rows(n=3, commodity="containers", delay_minutes=90)
        out = transformer._flag_sla_violations(df)
        assert not out["sla_violated"].any()

    def test_food_supplies_above_threshold(self, transformer):
        # food_supplies threshold is 60 min; delay=80 -> violated
        df = _make_shipment_rows(n=3, commodity="food_supplies", delay_minutes=80)
        out = transformer._flag_sla_violations(df)
        assert out["sla_violated"].all()

    def test_threshold_mapping_matches_constants(self, transformer):
        df = _make_shipment_rows(n=1, commodity="petrochemicals", delay_minutes=0)
        out = transformer._flag_sla_violations(df)
        assert out["sla_threshold_minutes"].iloc[0] == SLA_THRESHOLDS["petrochemicals"]


# ---------------------------------------------------------------------------
# _add_seasonal_flags
# ---------------------------------------------------------------------------

class TestSeasonalFlags:
    def test_hajj_flag_july(self, transformer):
        df = _make_shipment_rows(n=2, base_date=datetime(2023, 7, 10))
        out = transformer._add_seasonal_flags(df)
        assert out["is_hajj_season"].all()
        assert out["is_summer"].all()

    def test_no_hajj_in_january(self, transformer):
        df = _make_shipment_rows(n=2, base_date=datetime(2023, 1, 5))
        out = transformer._add_seasonal_flags(df)
        assert not out["is_hajj_season"].any()
        assert not out["is_ramadan"].any()

    def test_ramadan_flag(self, transformer):
        df = _make_shipment_rows(n=2, base_date=datetime(2023, 3, 20))
        out = transformer._add_seasonal_flags(df)
        assert out["is_ramadan"].all()

    def test_quarter_and_dow_present(self, transformer, raw_df):
        out = transformer._add_seasonal_flags(raw_df)
        assert "quarter" in out.columns
        assert "day_of_week" in out.columns
        assert "hour_of_day" in out.columns


# ---------------------------------------------------------------------------
# _add_route_column / _add_delay_category
# ---------------------------------------------------------------------------

class TestRouteAndCategory:
    def test_route_concatenation(self, transformer):
        df = _make_shipment_rows(n=1, origin="Jubail", destination="Dammam")
        out = transformer._add_route_column(df)
        assert out["route"].iloc[0] == "Jubail - Dammam"

    def test_delay_category_on_time(self, transformer, on_time_df):
        out = transformer._add_delay_category(on_time_df)
        assert (out["delay_category"] == "on_time").all()

    def test_delay_category_minor(self, transformer):
        df = _make_shipment_rows(n=1, delay_minutes=30)
        out = transformer._add_delay_category(df)
        assert out["delay_category"].iloc[0] == "minor_1h"

    def test_delay_category_major(self, transformer):
        df = _make_shipment_rows(n=1, delay_minutes=2000)
        out = transformer._add_delay_category(df)
        assert out["delay_category"].iloc[0] == "major_24h_plus"


# ---------------------------------------------------------------------------
# Full transform() pass
# ---------------------------------------------------------------------------

class TestFullTransform:
    def test_all_columns_present(self, transformed_df):
        expected = {
            "delay_hours", "departure_delay_hours", "sla_violated",
            "sla_threshold_minutes", "is_hajj_season", "is_ramadan",
            "is_summer", "quarter", "day_of_week", "hour_of_day",
            "route", "delay_category",
        }
        assert expected.issubset(set(transformed_df.columns))

    def test_row_count_unchanged(self, raw_df, transformed_df):
        assert len(transformed_df) == len(raw_df)

    def test_aggregations_populated(self, transformer, raw_df):
        transformer.transform(raw_df)
        assert transformer.get_route_performance() is not None
        assert transformer.get_commodity_analysis() is not None
        assert transformer.get_rolling_kpis() is not None


# ---------------------------------------------------------------------------
# Aggregation outputs
# ---------------------------------------------------------------------------

class TestAggregations:
    def test_route_performance_shape(self, transformer, mixed_commodity_df):
        transformer.transform(mixed_commodity_df)
        rp = transformer.get_route_performance()
        assert "route" in rp.columns
        assert "on_time_pct" in rp.columns
        assert "sla_compliance_pct" in rp.columns
        assert len(rp) >= 1

    def test_commodity_analysis_commodities(self, transformer, mixed_commodity_df):
        transformer.transform(mixed_commodity_df)
        ca = transformer.get_commodity_analysis()
        assert set(ca["commodity"]) == {"petrochemicals", "food_supplies", "minerals", "cement"}

    def test_rolling_kpis_has_rolling_cols(self, transformer, multi_day_df):
        transformer.transform(multi_day_df)
        kpis = transformer.get_rolling_kpis()
        assert "avg_delay_7d" in kpis.columns
        assert "avg_delay_30d" in kpis.columns
        assert "volume_7d" in kpis.columns
        assert len(kpis) > 0


# ---------------------------------------------------------------------------
# PipelineOrchestrator smoke test
# ---------------------------------------------------------------------------

class TestPipelineOrchestratorInit:
    """Basic instantiation checks -- no full run to avoid file I/O."""

    def test_default_mode(self):
        from src.pipeline_orchestrator import PipelineOrchestrator
        orch = PipelineOrchestrator()
        assert orch.mode == "full"
        assert orch.df is None
        assert orch.steps == []

    def test_incremental_mode(self):
        from src.pipeline_orchestrator import PipelineOrchestrator
        orch = PipelineOrchestrator(mode="incremental")
        assert orch.mode == "incremental"

    def test_pipeline_halt_error_is_exception(self):
        from src.pipeline_orchestrator import PipelineHaltError
        with pytest.raises(PipelineHaltError):
            raise PipelineHaltError("quality gate failed")

    def test_build_summary_structure(self):
        from src.pipeline_orchestrator import PipelineOrchestrator
        import time
        orch = PipelineOrchestrator()
        orch.start_time = time.time()
        orch.df = pd.DataFrame({"x": [1, 2, 3]})
        summary = orch._build_summary("SUCCESS")
        assert summary["status"] == "SUCCESS"
        assert summary["total_records"] == 3
        assert "total_runtime_seconds" in summary
