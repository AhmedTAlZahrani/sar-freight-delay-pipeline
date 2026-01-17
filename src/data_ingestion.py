import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
import time


EXPECTED_COLUMNS = [
    "shipment_id", "origin", "destination", "commodity",
    "weight_tons", "container_count", "scheduled_departure",
    "actual_departure", "scheduled_arrival", "actual_arrival",
    "delay_minutes", "delay_cause", "carrier_id", "customer_id",
]

SCHEMA = pa.schema([
    ("shipment_id", pa.string()),
    ("origin", pa.string()),
    ("destination", pa.string()),
    ("commodity", pa.string()),
    ("weight_tons", pa.float64()),
    ("container_count", pa.int64()),
    ("scheduled_departure", pa.timestamp("us")),
    ("actual_departure", pa.timestamp("us")),
    ("scheduled_arrival", pa.timestamp("us")),
    ("actual_arrival", pa.timestamp("us")),
    ("delay_minutes", pa.int64()),
    ("delay_cause", pa.string()),
    ("carrier_id", pa.string()),
    ("customer_id", pa.string()),
    ("year", pa.int32()),
    ("month", pa.int32()),
    ("day", pa.int32()),
])


class DataIngestor:
    """Ingest CSV freight data and convert to partitioned Parquet format.

    Follows S3 data lake patterns with year/month/day partitioning
    for efficient query performance with tools like AWS Athena or Glue.
    """

    def __init__(self, output_dir="data/parquet"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ingestion_log = []

    def ingest_csv(self, csv_path):
        """Read a CSV file and validate its schema.

        Args:
            csv_path: Path to the source CSV file.

        Returns:
            DataFrame with parsed datetime columns.
        """
        start = time.time()
        print(f"Ingesting CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        self._validate_schema(df)

        datetime_cols = [
            "scheduled_departure", "actual_departure",
            "scheduled_arrival", "actual_arrival",
        ]
        for col in datetime_cols:
            df[col] = pd.to_datetime(df[col])

        elapsed = time.time() - start
        self._log_ingestion(csv_path, len(df), elapsed, "csv_read")
        print(f"  Read {len(df):,} rows in {elapsed:.2f}s")
        return df

    def _validate_schema(self, df):
        """Validate that the DataFrame contains all expected columns.

        Args:
            df: DataFrame to validate.

        Raises:
            ValueError: If required columns are missing.
        """
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in source data: {missing}")

        extra = set(df.columns) - set(EXPECTED_COLUMNS)
        if extra:
            print(f"  Warning: unexpected columns will be kept: {extra}")

    def add_partition_columns(self, df):
        """Add year, month, day columns for Parquet partitioning.

        Args:
            df: DataFrame with scheduled_departure column.

        Returns:
            DataFrame with partition columns added.
        """
        df = df.copy()
        df["year"] = df["scheduled_departure"].dt.year
        df["month"] = df["scheduled_departure"].dt.month
        df["day"] = df["scheduled_departure"].dt.day
        return df

    def write_parquet(self, df, partition_cols=None):
        """Write DataFrame to partitioned Parquet files.

        Args:
            df: DataFrame to write.
            partition_cols: Columns to partition by. Defaults to year/month.

        Returns:
            Path to the output directory.
        """
        start = time.time()
        partition_cols = partition_cols or ["year", "month"]

        df = self.add_partition_columns(df)

        table = pa.Table.from_pandas(df)
        pq.write_to_dataset(
            table,
            root_path=str(self.output_dir),
            partition_cols=partition_cols,
        )

        elapsed = time.time() - start
        self._log_ingestion(str(self.output_dir), len(df), elapsed, "parquet_write")
        print(f"  Wrote {len(df):,} rows to Parquet in {elapsed:.2f}s")
        print(f"  Output: {self.output_dir}")
        return self.output_dir

    def read_parquet(self, filters=None):
        """Read Parquet dataset with optional partition filters.

        Args:
            filters: PyArrow filter expressions for partition pruning.

        Returns:
            DataFrame with filtered data.
        """
        start = time.time()
        dataset = pq.ParquetDataset(str(self.output_dir), filters=filters)
        df = dataset.read().to_pandas()
        elapsed = time.time() - start
        print(f"  Read {len(df):,} rows from Parquet in {elapsed:.2f}s")
        return df

    def ingest_and_convert(self, csv_path):
        """Full ingestion pipeline: CSV to partitioned Parquet.

        Args:
            csv_path: Path to source CSV.

        Returns:
            DataFrame after ingestion.
        """
        print("=" * 60)
        print("INGESTION PIPELINE")
        print("=" * 60)

        df = self.ingest_csv(csv_path)
        self.write_parquet(df)

        self._print_ingestion_summary()
        return df

    def _log_ingestion(self, source, row_count, elapsed, step):
        """Record ingestion step details.

        Args:
            source: Source path or identifier.
            row_count: Number of rows processed.
            elapsed: Time taken in seconds.
            step: Name of the ingestion step.
        """
        self.ingestion_log.append({
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "source": str(source),
            "row_count": row_count,
            "elapsed_seconds": round(elapsed, 3),
        })

    def _print_ingestion_summary(self):
        """Print a summary of all ingestion steps."""
        print("\nIngestion Summary:")
        print("-" * 50)
        for entry in self.ingestion_log:
            print(f"  [{entry['step']}] {entry['row_count']:,} rows | {entry['elapsed_seconds']:.3f}s")
        total_time = sum(e["elapsed_seconds"] for e in self.ingestion_log)
        print(f"  Total ingestion time: {total_time:.3f}s")

    def get_partition_stats(self):
        partitions = list(self.output_dir.glob("year=*/month=*"))
        return {
            "partition_count": len(partitions),
            "output_dir": str(self.output_dir),
        }

    def get_ingestion_log(self):
        """Return the ingestion log as a DataFrame.

        Returns:
            DataFrame with ingestion step details.
        """
        return pd.DataFrame(self.ingestion_log)

