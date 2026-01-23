import pandas as pd
import numpy as np


SLA_THRESHOLDS = {
    "petrochemicals": 120,
    "containers": 180,
    "minerals": 240,
    "cement": 180,
    "food_supplies": 60,
    "building_materials": 180,
}

HAJJ_MONTHS = [6, 7]
RAMADAN_MONTHS = [3, 4]
SUMMER_MONTHS = [6, 7, 8]


class DataTransformer:
    """Transform raw freight shipment data into analytical features.

    Calculates delay metrics, flags SLA violations, computes route
    performance aggregations, rolling KPIs, and seasonal adjustments.
    """

    def __init__(self):
        self.route_performance = None
        self.commodity_analysis = None
        self.rolling_kpis = None

    def transform(self, df):
        """Run all transformations on the shipment data.

        Args:
            df: Raw shipment DataFrame.

        Returns:
            Transformed DataFrame with derived columns.
        """
        print("=" * 60)
        print("DATA TRANSFORMATIONS")
        print("=" * 60)

        df = self._ensure_datetime(df)
        df = self._calculate_delay_hours(df)
        df = self._flag_sla_violations(df)
        df = self._add_seasonal_flags(df)
        df = self._add_route_column(df)
        df = self._add_delay_category(df)

        self.route_performance = self._aggregate_route_performance(df)
        self.commodity_analysis = self._aggregate_commodity_delays(df)
        self.rolling_kpis = self._compute_rolling_kpis(df)

        print("Processed {} records".format(len(df)))
        return df

    def _ensure_datetime(self, df):
        df = df.copy()
        datetime_cols = [
            "scheduled_departure", "actual_departure",
            "scheduled_arrival", "actual_arrival",
        ]
        for col in datetime_cols:
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col])
        return df

    def _calculate_delay_hours(self, df):
        """Calculate delay in hours from actual vs scheduled arrival.

        Args:
            df: DataFrame with arrival timestamps.

        Returns:
            DataFrame with delay_hours column added.
        """
        print("  Calculating delay hours...")
        df["delay_hours"] = (
            (df["actual_arrival"] - df["scheduled_arrival"]).dt.total_seconds() / 3600
        ).round(2)

        # Also compute departure delay
        df["departure_delay_hours"] = (
            (df["actual_departure"] - df["scheduled_departure"]).dt.total_seconds() / 3600
        ).round(2)

        avg_delay = df["delay_hours"].mean()
        print(f"    Average delay: {avg_delay:.2f} hours")
        return df

    def _flag_sla_violations(self, df):
        """Flag shipments that exceed commodity-specific SLA thresholds.

        Args:
            df: DataFrame with delay_minutes and commodity columns.

        Returns:
            DataFrame with sla_violated and sla_threshold columns.
        """
        print("  Flagging SLA violations...")
        df["sla_threshold_minutes"] = df["commodity"].map(SLA_THRESHOLDS)
        df["sla_violated"] = df["delay_minutes"] > df["sla_threshold_minutes"]

        violation_rate = df["sla_violated"].mean()
        print(f"    SLA violation rate: {violation_rate:.1%}")
        return df

    def _add_seasonal_flags(self, df):
        """Add flags for Hajj season, Ramadan, and summer periods.

        Args:
            df: DataFrame with scheduled_departure column.

        Returns:
            DataFrame with seasonal flag columns.
        """
        print("  Adding seasonal flags...")
        month = df["scheduled_departure"].dt.month

        df["is_hajj_season"] = month.isin(HAJJ_MONTHS)
        df["is_ramadan"] = month.isin(RAMADAN_MONTHS)
        df["is_summer"] = month.isin(SUMMER_MONTHS)
        df["quarter"] = df["scheduled_departure"].dt.quarter
        df["day_of_week"] = df["scheduled_departure"].dt.dayofweek
        df["hour_of_day"] = df["scheduled_departure"].dt.hour

        hajj_pct = df["is_hajj_season"].mean()
        print(f"    Hajj season records: {hajj_pct:.1%}")
        return df

    def _add_route_column(self, df):
        df["route"] = df["origin"] + " - " + df["destination"]
        return df

    def _add_delay_category(self, df):
        """Categorize delays into severity buckets.

        Args:
            df: DataFrame with delay_minutes column.

        Returns:
            DataFrame with delay_category column.
        """
        print("  Categorizing delays...")
        conditions = [
            df["delay_minutes"] == 0,
            df["delay_minutes"].between(1, 60),
            df["delay_minutes"].between(61, 240),
            df["delay_minutes"].between(241, 1440),
            df["delay_minutes"] > 1440,
        ]
        labels = ["on_time", "minor_1h", "moderate_1_4h", "significant_4_24h", "major_24h_plus"]
        df["delay_category"] = np.select(conditions, labels, default="unknown")

        for label in labels:
            pct = (df["delay_category"] == label).mean()
            print(f"    {label}: {pct:.1%}")
        return df

    def _aggregate_route_performance(self, df):
        """Compute route-level performance metrics.

        Args:
            df: Transformed DataFrame.

        Returns:
            DataFrame with route performance summary.
        """
        print("  Aggregating route performance...")
        route_perf = df.groupby("route").agg(
            total_shipments=("shipment_id", "count"),
            avg_delay_minutes=("delay_minutes", "mean"),
            median_delay_minutes=("delay_minutes", "median"),
            max_delay_minutes=("delay_minutes", "max"),
            on_time_count=("delay_minutes", lambda x: (x == 0).sum()),
            sla_violations=("sla_violated", "sum"),
            avg_weight_tons=("weight_tons", "mean"),
            total_weight_tons=("weight_tons", "sum"),
        ).reset_index()

        route_perf["on_time_pct"] = (
            route_perf["on_time_count"] / route_perf["total_shipments"] * 100
        ).round(1)
        route_perf["sla_compliance_pct"] = (
            (1 - route_perf["sla_violations"] / route_perf["total_shipments"]) * 100
        ).round(1)
        route_perf["avg_delay_minutes"] = route_perf["avg_delay_minutes"].round(1)

        print(f"    Routes analyzed: {len(route_perf)}")
        return route_perf

    def _aggregate_commodity_delays(self, df):
        """Compute commodity-level delay analysis.

        Args:
            df: Transformed DataFrame.

        Returns:
            DataFrame with commodity delay summary.
        """
        print("  Analyzing commodity delays...")
        commodity_stats = df.groupby("commodity").agg(
            total_shipments=("shipment_id", "count"),
            avg_delay_minutes=("delay_minutes", "mean"),
            median_delay_minutes=("delay_minutes", "median"),
            sla_violations=("sla_violated", "sum"),
            sla_threshold=("sla_threshold_minutes", "first"),
            avg_weight=("weight_tons", "mean"),
        ).reset_index()

        commodity_stats["sla_compliance_pct"] = (
            (1 - commodity_stats["sla_violations"] / commodity_stats["total_shipments"]) * 100
        ).round(1)
        commodity_stats["avg_delay_minutes"] = commodity_stats["avg_delay_minutes"].round(1)

        return commodity_stats

    def _compute_rolling_kpis(self, df):
        """Calculate 7-day and 30-day rolling average delays.

        Args:
            df: Transformed DataFrame.

        Returns:
            DataFrame with daily KPIs and rolling averages.
        """
        print("  Computing rolling KPIs...")
        daily = df.set_index("scheduled_departure").resample("D").agg(
            shipment_count=("shipment_id", "count"),
            avg_delay=("delay_minutes", "mean"),
            on_time_count=("delay_minutes", lambda x: (x == 0).sum()),
            sla_violations=("sla_violated", "sum"),
        ).reset_index()

        daily["on_time_pct"] = (
            daily["on_time_count"] / daily["shipment_count"].replace(0, np.nan) * 100
        ).round(1)

        daily["avg_delay_7d"] = daily["avg_delay"].rolling(7, min_periods=1).mean().round(1)
        daily["avg_delay_30d"] = daily["avg_delay"].rolling(30, min_periods=1).mean().round(1)
        daily["on_time_pct_7d"] = daily["on_time_pct"].rolling(7, min_periods=1).mean().round(1)
        daily["on_time_pct_30d"] = daily["on_time_pct"].rolling(30, min_periods=1).mean().round(1)
        daily["volume_7d"] = daily["shipment_count"].rolling(7, min_periods=1).sum()
        daily["volume_30d"] = daily["shipment_count"].rolling(30, min_periods=1).sum()

        print(f"    Daily KPIs computed: {len(daily)} days")
        return daily

    # TODO: add retry logic for API timeouts
    def get_route_performance(self):
        """Return the route performance aggregation."""
        return self.route_performance

    def get_commodity_analysis(self):
        """Return the commodity delay analysis.

        Returns:
            DataFrame with commodity-level metrics.
        """
        return self.commodity_analysis

    def get_rolling_kpis(self):
        """Return the rolling KPI DataFrame.

        Returns:
            DataFrame with daily and rolling metrics.
        """
        return self.rolling_kpis
