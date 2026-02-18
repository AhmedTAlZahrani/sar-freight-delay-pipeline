import json
import time
from pathlib import Path
from datetime import datetime

import pandas as pd

from .data_generator import FreightDataGenerator
from .data_ingestion import DataIngestor
from .data_quality import DataQualityChecker
from .transformations import DataTransformer
from .reporting import ReportGenerator


PIPELINE_OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "freight_shipments.csv"
PARQUET_DIR = DATA_DIR / "parquet"


class PipelineOrchestrator:
    """Orchestrate the SAR freight delay data pipeline.

    Executes a DAG-style pipeline: generate/ingest -> validate ->
    transform -> aggregate -> report. Supports full refresh and
    incremental modes with quality gate enforcement.
    """

    def __init__(self, mode="full"):
        self.mode = mode
        self.steps = []
        self.start_time = None
        self.df = None
        self.quality_report = None
        self.transformer = None

        PIPELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Execute the full pipeline DAG.

        Returns:
            Dict with pipeline summary including timing and status.
        """
        self.start_time = time.time()

        print("=" * 60)
        print(f"SAR FREIGHT DELAY PIPELINE ({self.mode.upper()} REFRESH)")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        try:
            self._step_generate()
            self._step_ingest()
            self._step_validate()
            self._step_transform()
            self._step_aggregate()
            self._step_report()
        except PipelineHaltError as e:
            print(f"\nPIPELINE HALTED: {e}")
            self._save_pipeline_state("HALTED")
            return self._build_summary("HALTED")

        self._save_pipeline_state("SUCCESS")
        summary = self._build_summary("SUCCESS")
        self._print_summary(summary)
        return summary

    def _step_generate(self):
        step_start = time.time()

        if self.mode == "full" or not CSV_PATH.exists():
            print("\n[Step 1/6] Generating freight data...")
            generator = FreightDataGenerator()
            self.df = generator.generate()
            generator.save(self.df, str(CSV_PATH))
        else:
            print("\n[Step 1/6] Skipping generation (incremental mode)...")
            self.df = pd.read_csv(CSV_PATH)

        self._log_step("generate", time.time() - step_start, len(self.df))

    def _step_ingest(self):
        step_start = time.time()
        print("\n[Step 2/6] Ingesting data...")

        ingestor = DataIngestor(output_dir=str(PARQUET_DIR))
        self.df = ingestor.ingest_csv(str(CSV_PATH))
        ingestor.write_parquet(self.df)

        self._log_step("ingest", time.time() - step_start, len(self.df))

    def _step_validate(self):
        step_start = time.time()
        print("\n[Step 3/6] Running quality checks...")

        checker = DataQualityChecker()
        self.quality_report = checker.run_all_checks(self.df)

        if checker.should_halt():
            self._log_step("validate", time.time() - step_start, 0)
            raise PipelineHaltError(
                f"Quality gate failed with {len(checker.critical_failures)} critical issues"
            )

        # Save quality report
        quality_df = checker.get_report_df()
        quality_df.to_csv(PIPELINE_OUTPUT_DIR / "quality_report.csv", index=False)

        self._log_step("validate", time.time() - step_start, len(self.df))

    def _step_transform(self):
        """Apply delay transformations and SLA flagging."""
        step_start = time.time()
        print("\n[Step 4/6] Transforming data...")

        self.transformer = DataTransformer()
        self.df = self.transformer.transform(self.df)

        # Save transformed data
        self.df.to_parquet(PIPELINE_OUTPUT_DIR / "transformed_shipments.parquet", index=False)

        self._log_step("transform", time.time() - step_start, len(self.df))

    def _step_aggregate(self):
        step_start = time.time()
        print("\n[Step 5/6] Aggregating data...")

        route_perf = self.transformer.get_route_performance()
        commodity = self.transformer.get_commodity_analysis()
        rolling = self.transformer.get_rolling_kpis()

        route_perf.to_csv(PIPELINE_OUTPUT_DIR / "route_performance.csv", index=False)
        commodity.to_csv(PIPELINE_OUTPUT_DIR / "commodity_analysis.csv", index=False)
        rolling.to_csv(PIPELINE_OUTPUT_DIR / "rolling_kpis.csv", index=False)

        print(f"  Saved route performance: {len(route_perf)} routes")
        print(f"  Saved commodity analysis: {len(commodity)} commodities")
        print(f"  Saved rolling KPIs: {len(rolling)} daily records")

        self._log_step("aggregate", time.time() - step_start, len(self.df))

    def _step_report(self):
        step_start = time.time()
        print("\n[Step 6/6] Generating reports...")

        reporter = ReportGenerator(output_dir=str(PIPELINE_OUTPUT_DIR))
        reporter.generate_summary(self.df)
        reporter.generate_delay_distribution(self.df)
        reporter.generate_route_cards(self.transformer.get_route_performance())
        reporter.generate_delay_cause_pareto(self.df)
        reporter.export_all()

        self._log_step("report", time.time() - step_start, len(self.df))

    def _log_step(self, name, elapsed, row_count):
        """Record pipeline step execution details.

        Args:
            name: Step name.
            elapsed: Elapsed time in seconds.
            row_count: Number of rows processed.
        """
        self.steps.append({
            "step": name,
            "elapsed_seconds": round(elapsed, 3),
            "row_count": row_count,
            "timestamp": datetime.now().isoformat(),
        })
        print(f"  Step '{name}' completed in {elapsed:.2f}s")

    def _build_summary(self, status):
        """Build a pipeline execution summary.

        Args:
            status: Final pipeline status string.

        Returns:
            Dict with execution summary.
        """
        total_time = time.time() - self.start_time
        return {
            "status": status,
            "mode": self.mode,
            "total_runtime_seconds": round(total_time, 2),
            "total_records": len(self.df) if self.df is not None else 0,
            "steps": self.steps,
            "quality_pass_rate": self.quality_report.get("pass_rate") if self.quality_report else None,
            "completed_at": datetime.now().isoformat(),
        }

    def _print_summary(self, summary):
        """Print the final pipeline summary.

        Args:
            summary: Pipeline summary dict.
        """
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"  Status:          {summary['status']}")
        print(f"  Mode:            {summary['mode']}")
        print(f"  Total runtime:   {summary['total_runtime_seconds']:.2f}s")
        print(f"  Records:         {summary['total_records']:,}")
        print(f"  Quality rate:    {summary['quality_pass_rate']}%")
        print(f"  Completed:       {summary['completed_at']}")
        print("\n  Step Timings:")
        for step in summary["steps"]:
            print(f"    {step['step']:15s} {step['elapsed_seconds']:8.3f}s")

    def _save_pipeline_state(self, status):
        """Save pipeline execution state to JSON.

        Args:
            status: Final pipeline status.
        """
        state = self._build_summary(status)
        state_path = PIPELINE_OUTPUT_DIR / "pipeline_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        print(f"\nPipeline state saved to {state_path}")


class PipelineHaltError(Exception):
    """Raised when a quality gate fails and the pipeline must stop."""
    pass


if __name__ == "__main__":
    orchestrator = PipelineOrchestrator(mode="full")
    orchestrator.run()

