import json
import pandas as pd
import numpy as np
from pathlib import Path


TOP_N_CAUSES = 8
PARETO_THRESHOLD = 80.0


class ReportGenerator:
    """Generate analytical reports for SAR freight delay data.

    Produces summary statistics, delay distributions, route performance
    cards, and delay cause Pareto analysis. Exports to CSV and JSON.
    """

    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports = {}

    def generate_summary(self, df):
        """Generate high-level summary statistics.

        Args:
            df: Transformed shipment DataFrame.

        Returns:
            Dict with summary statistics.
        """
        print("  Generating summary statistics...")

        total = len(df)
        on_time = (df["delay_minutes"] == 0).sum()
        delayed = total - on_time

        summary = {
            "total_shipments": total,
            "on_time_shipments": int(on_time),
            "delayed_shipments": int(delayed),
            "on_time_rate": round(on_time / total * 100, 1),
            "avg_delay_minutes": round(df["delay_minutes"].mean(), 1),
            "median_delay_minutes": round(df["delay_minutes"].median(), 1),
            "max_delay_minutes": int(df["delay_minutes"].max()),
            "p95_delay_minutes": round(df["delay_minutes"].quantile(0.95), 1),
            "p99_delay_minutes": round(df["delay_minutes"].quantile(0.99), 1),
            "total_weight_tons": round(df["weight_tons"].sum(), 1),
            "unique_routes": df["route"].nunique() if "route" in df.columns else 0,
            "unique_carriers": df["carrier_id"].nunique(),
            "unique_customers": df["customer_id"].nunique(),
            "date_range_start": str(df["scheduled_departure"].min()),
            "date_range_end": str(df["scheduled_departure"].max()),
        }

        if "sla_violated" in df.columns:
            sla_violations = df["sla_violated"].sum()
            summary["sla_violations"] = int(sla_violations)
            summary["sla_compliance_rate"] = round(
                (1 - sla_violations / total) * 100, 1
            )

        self.reports["summary"] = summary
        print(f"    Total: {total:,} | On-time: {summary['on_time_rate']}% | Avg delay: {summary['avg_delay_minutes']} min")
        return summary

    def generate_delay_distribution(self, df):
        """Analyze the distribution of delays across severity categories.

        Args:
            df: Transformed shipment DataFrame.

        Returns:
            DataFrame with delay distribution breakdown.
        """
        print("  Generating delay distribution...")

        bins = [0, 1, 60, 240, 1440, float("inf")]
        labels = ["On Time", "Under 1hr", "1-4 hrs", "4-24 hrs", "24+ hrs"]
        df_copy = df.copy()
        df_copy["delay_bucket"] = pd.cut(
            df_copy["delay_minutes"], bins=bins, labels=labels, right=True,
            include_lowest=True,
        )

        distribution = df_copy["delay_bucket"].value_counts().sort_index()
        dist_df = pd.DataFrame({
            "category": distribution.index.astype(str),
            "count": distribution.values,
            "percentage": (distribution.values / len(df_copy) * 100).round(1),
        })
        dist_df["cumulative_pct"] = dist_df["percentage"].cumsum().round(1)

        self.reports["delay_distribution"] = dist_df
        return dist_df

    def generate_route_cards(self, route_performance):
        """Create performance summary cards for each route.

        Args:
            route_performance: Route-level aggregation DataFrame.

        Returns:
            List of route performance card dicts.
        """
        print("  Generating route performance cards...")

        cards = []
        for _, row in route_performance.iterrows():
            card = {
                "route": row["route"],
                "total_shipments": int(row["total_shipments"]),
                "avg_delay_minutes": round(row["avg_delay_minutes"], 1),
                "on_time_pct": round(row["on_time_pct"], 1),
                "sla_compliance_pct": round(row["sla_compliance_pct"], 1),
                "total_weight_tons": round(row["total_weight_tons"], 1),
                "max_delay_minutes": int(row["max_delay_minutes"]),
            }
            cards.append(card)
            print(f"    {card['route']}: {card['total_shipments']:,} shipments | On-time: {card['on_time_pct']}%")

        self.reports["route_cards"] = cards
        return cards

    def generate_delay_cause_pareto(self, df):
        """Perform Pareto analysis (80/20 rule) on delay causes.

        Args:
            df: Transformed shipment DataFrame.

        Returns:
            DataFrame with Pareto analysis of delay causes.
        """
        print("  Generating delay cause Pareto analysis...")

        delayed = df[df["delay_cause"] != "none"].copy()

        cause_counts = delayed["delay_cause"].value_counts().reset_index()
        cause_counts.columns = ["delay_cause", "count"]
        cause_counts["percentage"] = (cause_counts["count"] / cause_counts["count"].sum() * 100).round(1)
        cause_counts["cumulative_pct"] = cause_counts["percentage"].cumsum().round(1)

        cause_counts["in_pareto_80"] = cause_counts["cumulative_pct"] <= PARETO_THRESHOLD

        # Add average delay per cause
        avg_delay_per_cause = delayed.groupby("delay_cause")["delay_minutes"].mean().round(1)
        cause_counts["avg_delay_minutes"] = cause_counts["delay_cause"].map(avg_delay_per_cause)

        # Add total delay impact
        total_delay_per_cause = delayed.groupby("delay_cause")["delay_minutes"].sum()
        cause_counts["total_delay_minutes"] = cause_counts["delay_cause"].map(total_delay_per_cause)
        cause_counts["delay_impact_pct"] = (
            cause_counts["total_delay_minutes"] / cause_counts["total_delay_minutes"].sum() * 100
        ).round(1)

        self.reports["pareto"] = cause_counts

        pareto_causes = cause_counts[cause_counts["in_pareto_80"]]["delay_cause"].tolist()
        print(f"    Top causes (80% of delays): {', '.join(pareto_causes)}")
        return cause_counts

    def export_all(self):
        """Export all generated reports to CSV and JSON files."""
        print("  Exporting reports...")

        # Export summary as JSON
        if "summary" in self.reports:
            summary_path = self.output_dir / "summary_statistics.json"
            with open(summary_path, "w") as f:
                json.dump(self.reports["summary"], f, indent=2, default=str)
            print(f"    Saved: {summary_path}")

        # Export delay distribution as CSV
        if "delay_distribution" in self.reports:
            dist_path = self.output_dir / "delay_distribution.csv"
            self.reports["delay_distribution"].to_csv(dist_path, index=False)
            print(f"    Saved: {dist_path}")

        # Export route cards as JSON
        if "route_cards" in self.reports:
            cards_path = self.output_dir / "route_cards.json"
            with open(cards_path, "w") as f:
                json.dump(self.reports["route_cards"], f, indent=2, default=str)
            print(f"    Saved: {cards_path}")

        # Export Pareto analysis as CSV
        if "pareto" in self.reports:
            pareto_path = self.output_dir / "delay_cause_pareto.csv"
            self.reports["pareto"].to_csv(pareto_path, index=False)
            print(f"    Saved: {pareto_path}")

        # Export combined report as JSON
        combined = {}
        for key, val in self.reports.items():
            if isinstance(val, pd.DataFrame):
                combined[key] = val.to_dict(orient="records")
            elif isinstance(val, list):
                combined[key] = val
            else:
                combined[key] = val

        combined_path = self.output_dir / "full_report.json"
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"    Saved: {combined_path}")

    def get_report(self, name):
        """Retrieve a specific report by name.

        Args:
            name: Report name (summary, delay_distribution, route_cards, pareto).

        Returns:
            Report data (dict, DataFrame, or list).
        """
        return self.reports.get(name)

    def get_all_reports(self):
        """Return all generated reports.

        Returns:
            Dict with all report data.
        """
        return self.reports
