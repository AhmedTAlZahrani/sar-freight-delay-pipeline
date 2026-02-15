# SAR Freight Delay Pipeline

Data pipeline for analyzing freight shipment delays on the Saudi Arabia Railways (SAR) network. Generates synthetic shipment data, runs quality checks, computes delay metrics and SLA flags, and serves an interactive Streamlit dashboard. Built around AWS patterns (S3 partitioned Parquet, Glue, CloudWatch) but runs locally for development.

## Install

```bash
git clone https://github.com/AhmedTAlZahrani/sar-freight-delay-pipeline.git
cd sar-freight-delay-pipeline
pip install -r requirements.txt
```

## Run

```bash
# generate 200K synthetic freight records
python -m src.data_generator

# run the full pipeline (ingest, validate, transform, report)
python -m src.pipeline_orchestrator

# launch the dashboard
streamlit run app.py
```

## Project Structure

```
src/
  data_generator.py        - synthetic freight data (200K records, 2022-2024)
  data_ingestion.py        - CSV to Parquet with year/month partitioning
  data_quality.py          - null checks, range validation, referential integrity
  transformations.py       - delay hours, SLA flags, route aggregation, rolling KPIs
  pipeline_orchestrator.py - DAG-style pipeline with quality gates
  reporting.py             - summary stats, Pareto analysis, route cards
app.py                     - Streamlit dashboard (Plotly dark theme)
```

## License

Apache License 2.0 -- see [LICENSE](LICENSE) for details.
