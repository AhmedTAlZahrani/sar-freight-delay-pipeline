import json
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")

PLOTLY_TEMPLATE = "plotly_dark"
COLOR_SEQUENCE = px.colors.qualitative.Set2


def load_pipeline_state():
    path = OUTPUT_DIR / "pipeline_state.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_summary_statistics():
    path = OUTPUT_DIR / "summary_statistics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_quality_report():
    path = OUTPUT_DIR / "quality_report.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_route_performance():
    """Load route performance aggregation.

    Returns:
        DataFrame with route metrics.
    """
    path = OUTPUT_DIR / "route_performance.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_commodity_analysis():
    """Load commodity delay analysis.

    Returns:
        DataFrame with commodity metrics.
    """
    path = OUTPUT_DIR / "commodity_analysis.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_rolling_kpis():
    """Load rolling KPI data.

    Returns:
        DataFrame with daily and rolling metrics.
    """
    path = OUTPUT_DIR / "rolling_kpis.csv"
    if path.exists():
        df = pd.read_csv(path)
        if "scheduled_departure" in df.columns:
            df["scheduled_departure"] = pd.to_datetime(df["scheduled_departure"])
        return df
    return pd.DataFrame()


def load_delay_distribution():
    """Load delay distribution data.

    Returns:
        DataFrame with delay distribution breakdown.
    """
    path = OUTPUT_DIR / "delay_distribution.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_pareto_data():
    """Load delay cause Pareto analysis.

    Returns:
        DataFrame with Pareto data.
    """
    path = OUTPUT_DIR / "delay_cause_pareto.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_transformed_data():
    """Load the transformed shipment data.

    Returns:
        DataFrame with transformed records.
    """
    path = OUTPUT_DIR / "transformed_shipments.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def render_sidebar(df):
    """Render sidebar filters for date range and route selection.

    Args:
        df: Transformed shipment DataFrame for filter options.

    Returns:
        Tuple of (start_date, end_date, selected_routes).
    """
    st.sidebar.header("Filters")

    if df.empty:
        return None, None, []

    if "scheduled_departure" in df.columns:
        min_date = df["scheduled_departure"].min().date()
        max_date = df["scheduled_departure"].max().date()
        start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    else:
        start_date, end_date = None, None

    if "route" in df.columns:
        routes = sorted(df["route"].unique().tolist())
        selected_routes = st.sidebar.multiselect("Routes", routes, default=routes)
    else:
        selected_routes = []

    return start_date, end_date, selected_routes


def filter_data(df, start_date, end_date, selected_routes):
    """Apply sidebar filters to the DataFrame.

    Args:
        df: Transformed shipment DataFrame.
        start_date: Start date filter.
        end_date: End date filter.
        selected_routes: List of selected routes.

    Returns:
        Filtered DataFrame.
    """
    if df.empty:
        return df

    filtered = df.copy()
    if start_date and end_date and "scheduled_departure" in filtered.columns:
        filtered = filtered[
            (filtered["scheduled_departure"].dt.date >= start_date)
            & (filtered["scheduled_departure"].dt.date <= end_date)
        ]
    if selected_routes and "route" in filtered.columns:
        filtered = filtered[filtered["route"].isin(selected_routes)]

    return filtered


def tab_pipeline_status(state, summary, quality_df):
    """Render the Pipeline Status tab.

    Args:
        state: Pipeline state dict.
        summary: Summary statistics dict.
        quality_df: Quality report DataFrame.
    """
    st.header("Pipeline Status")

    if not state:
        st.warning("No pipeline run detected. Run the pipeline first.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", state.get("status", "N/A"))
    col2.metric("Runtime", f"{state.get('total_runtime_seconds', 0):.1f}s")
    col3.metric("Records", f"{state.get('total_records', 0):,}")
    col4.metric("Quality Rate", f"{state.get('quality_pass_rate', 0)}%")

    st.subheader("Step Timings")
    if "steps" in state:
        steps_df = pd.DataFrame(state["steps"])
        fig = px.bar(
            steps_df, x="step", y="elapsed_seconds",
            title="Pipeline Step Duration",
            color="step", color_discrete_sequence=COLOR_SEQUENCE,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    if not quality_df.empty:
        st.subheader("Data Quality Checks")
        status_counts = quality_df["status"].value_counts()
        fig = px.pie(
            values=status_counts.values, names=status_counts.index,
            title="Quality Check Results",
            color_discrete_sequence=COLOR_SEQUENCE,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(quality_df, use_container_width=True)


def tab_route_performance(route_df, rolling_df):
    """Render the Route Performance tab.

    Args:
        route_df: Route performance DataFrame.
        rolling_df: Rolling KPIs DataFrame.
    """
    st.header("Route Performance")

    if route_df.empty:
        st.warning("No route performance data available.")
        return

    # KPI cards per route
    for _, row in route_df.iterrows():
        with st.expander(f"{row['route']} ({int(row['total_shipments']):,} shipments)", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("On-Time %", f"{row['on_time_pct']}%")
            c2.metric("Avg Delay", f"{row['avg_delay_minutes']:.0f} min")
            c3.metric("SLA Compliance", f"{row['sla_compliance_pct']}%")
            c4.metric("Total Weight", f"{row['total_weight_tons']:,.0f} t")

    # Route comparison chart
    fig = px.bar(
        route_df.sort_values("on_time_pct"),
        x="on_time_pct", y="route", orientation="h",
        title="On-Time Performance by Route",
        color="on_time_pct",
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Rolling KPIs trend
    if not rolling_df.empty and "scheduled_departure" in rolling_df.columns:
        st.subheader("Delay Trends (Rolling Averages)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_df["scheduled_departure"], y=rolling_df["avg_delay_7d"],
            mode="lines", name="7-day Avg Delay",
        ))
        fig.add_trace(go.Scatter(
            x=rolling_df["scheduled_departure"], y=rolling_df["avg_delay_30d"],
            mode="lines", name="30-day Avg Delay",
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=400,
            title="Rolling Average Delay (minutes)",
            xaxis_title="Date", yaxis_title="Avg Delay (min)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Volume Trends")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_df["scheduled_departure"], y=rolling_df["volume_30d"],
            mode="lines", name="30-day Volume", fill="tozeroy",
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=350,
            title="30-Day Rolling Shipment Volume",
            xaxis_title="Date", yaxis_title="Shipments",
        )
        st.plotly_chart(fig, use_container_width=True)


def tab_delay_analysis(dist_df, pareto_df, df):
    """Render the Delay Analysis tab.

    Args:
        dist_df: Delay distribution DataFrame.
        pareto_df: Pareto analysis DataFrame.
        df: Filtered shipment DataFrame.
    """
    st.header("Delay Analysis")

    if dist_df.empty:
        st.warning("No delay distribution data available.")
        return

    # Distribution chart
    fig = px.bar(
        dist_df, x="category", y="count",
        title="Delay Distribution",
        color="category", color_discrete_sequence=COLOR_SEQUENCE,
        text="percentage",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Pareto chart
    if not pareto_df.empty:
        st.subheader("Delay Cause Pareto Analysis")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pareto_df["delay_cause"], y=pareto_df["count"],
            name="Count", marker_color="#2ecc71",
        ))
        fig.add_trace(go.Scatter(
            x=pareto_df["delay_cause"], y=pareto_df["cumulative_pct"],
            mode="lines+markers", name="Cumulative %",
            yaxis="y2", marker_color="#e74c3c",
        ))
        fig.add_hline(y=80, line_dash="dash", line_color="yellow",
                      annotation_text="80% threshold", yref="y2")
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=450,
            title="Delay Cause Pareto (80/20 Rule)",
            yaxis=dict(title="Count"),
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Delay by cause - average impact
    if not pareto_df.empty:
        st.subheader("Average Delay by Cause")
        fig = px.bar(
            pareto_df.sort_values("avg_delay_minutes", ascending=True),
            x="avg_delay_minutes", y="delay_cause", orientation="h",
            title="Average Delay Duration by Cause",
            color="avg_delay_minutes", color_continuous_scale="Reds",
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=400)
        st.plotly_chart(fig, use_container_width=True)


def tab_sla_compliance(commodity_df, df):
    """Render the SLA Compliance tab.

    Args:
        commodity_df: Commodity analysis DataFrame.
        df: Filtered shipment DataFrame.
    """
    st.header("SLA Compliance")

    if commodity_df.empty:
        st.warning("No commodity analysis data available.")
        return

    # Commodity SLA compliance
    st.subheader("SLA Compliance by Commodity")
    fig = px.bar(
        commodity_df.sort_values("sla_compliance_pct"),
        x="sla_compliance_pct", y="commodity", orientation="h",
        title="SLA Compliance Rate by Commodity",
        color="sla_compliance_pct",
        color_continuous_scale="RdYlGn",
        text="sla_compliance_pct",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Commodity KPI cards
    for _, row in commodity_df.iterrows():
        with st.expander(f"{row['commodity']} (SLA threshold: {int(row['sla_threshold'])} min)"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Shipments", f"{int(row['total_shipments']):,}")
            c2.metric("SLA Compliance", f"{row['sla_compliance_pct']}%")
            c3.metric("Avg Delay", f"{row['avg_delay_minutes']:.0f} min")
            c4.metric("SLA Violations", f"{int(row['sla_violations']):,}")

    # Route-level SLA if we have the filtered data
    if not df.empty and "route" in df.columns and "sla_violated" in df.columns:
        st.subheader("SLA Compliance by Route")
        route_sla = df.groupby("route").agg(
            total=("shipment_id", "count"),
            violations=("sla_violated", "sum"),
        ).reset_index()
        route_sla["compliance_pct"] = ((1 - route_sla["violations"] / route_sla["total"]) * 100).round(1)

        fig = px.bar(
            route_sla.sort_values("compliance_pct"),
            x="compliance_pct", y="route", orientation="h",
            title="SLA Compliance Rate by Route",
            color="compliance_pct",
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Monthly SLA trend
    if not df.empty and "scheduled_departure" in df.columns and "sla_violated" in df.columns:
        st.subheader("Monthly SLA Compliance Trend")
        df_temp = df.copy()
        df_temp["month"] = df_temp["scheduled_departure"].dt.to_period("M").astype(str)
        monthly_sla = df_temp.groupby("month").agg(
            total=("shipment_id", "count"),
            violations=("sla_violated", "sum"),
        ).reset_index()
        monthly_sla["compliance_pct"] = ((1 - monthly_sla["violations"] / monthly_sla["total"]) * 100).round(1)

        fig = px.line(
            monthly_sla, x="month", y="compliance_pct",
            title="Monthly SLA Compliance Rate",
            markers=True,
        )
        fig.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="Target: 85%")
        fig.update_layout(template=PLOTLY_TEMPLATE, height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="SAR Freight Delay Pipeline",
        page_icon="🚂",
        layout="wide",
    )

    st.title("SAR Freight Delay Pipeline")
    st.caption("Data Engineering Pipeline for Saudi Arabia Railways Freight Shipment Analysis")

    # Load all data
    state = load_pipeline_state()
    summary = load_summary_statistics()
    quality_df = load_quality_report()
    route_df = load_route_performance()
    commodity_df = load_commodity_analysis()
    rolling_df = load_rolling_kpis()
    dist_df = load_delay_distribution()
    pareto_df = load_pareto_data()
    df = load_transformed_data()

    # Sidebar filters
    start_date, end_date, selected_routes = render_sidebar(df)
    filtered_df = filter_data(df, start_date, end_date, selected_routes)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Pipeline Status",
        "Route Performance",
        "Delay Analysis",
        "SLA Compliance",
    ])

    with tab1:
        tab_pipeline_status(state, summary, quality_df)

    with tab2:
        tab_route_performance(route_df, rolling_df)

    with tab3:
        tab_delay_analysis(dist_df, pareto_df, filtered_df)

    with tab4:
        tab_sla_compliance(commodity_df, filtered_df)


if __name__ == "__main__":
    main()

