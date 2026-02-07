import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


NUM_RECORDS = 200_000
DATE_START = datetime(2022, 1, 1)
DATE_END = datetime(2024, 12, 31)

ROUTES = [
    ("Riyadh", "Dammam"),
    ("Dammam", "Riyadh"),
    ("Riyadh", "Jubail"),
    ("Jubail", "Riyadh"),
    ("Dammam", "Jubail"),
    ("Jubail", "Dammam"),
    ("Riyadh", "Sudair"),
    ("Sudair", "Riyadh"),
]

MINING_CORRIDORS = [
    ("Ras Al Khair", "Jubail"),
    ("Jubail", "Ras Al Khair"),
]

ALL_ROUTES = ROUTES + MINING_CORRIDORS

COMMODITIES = [
    "petrochemicals",
    "containers",
    "minerals",
    "cement",
    "food_supplies",
    "building_materials",
]

DELAY_CAUSES = [
    "weather",
    "maintenance",
    "port_congestion",
    "customs_clearance",
    "loading_delay",
    "track_maintenance",
    "signal_failure",
    "none",
]

CARRIERS = [f"SAR-C{str(i).zfill(3)}" for i in range(1, 21)]
CUSTOMERS = [f"CUST-{str(i).zfill(4)}" for i in range(1, 151)]

ROUTE_DURATIONS = {
    ("Riyadh", "Dammam"): 240,
    ("Dammam", "Riyadh"): 240,
    ("Riyadh", "Jubail"): 300,
    ("Jubail", "Riyadh"): 300,
    ("Dammam", "Jubail"): 90,
    ("Jubail", "Dammam"): 90,
    ("Riyadh", "Sudair"): 120,
    ("Sudair", "Riyadh"): 120,
    ("Ras Al Khair", "Jubail"): 60,
    ("Jubail", "Ras Al Khair"): 60,
}

COMMODITY_WEIGHTS = {
    "petrochemicals": (50, 200),
    "containers": (20, 80),
    "minerals": (100, 500),
    "cement": (80, 300),
    "food_supplies": (10, 60),
    "building_materials": (40, 150),
}


class FreightDataGenerator:
    """Generate synthetic SAR freight shipment data with realistic patterns.

    Produces 200,000 records spanning 2022-2024 with Saudi-specific
    seasonal patterns including Hajj surges, Ramadan slowdowns,
    and summer heat maintenance windows.
    """

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.records = []

    def generate(self):
        """Generate all freight shipment records.

        Returns:
            DataFrame with 200,000 synthetic shipment records.
        """
        print("Generating SAR freight shipment data...")
        total_days = (DATE_END - DATE_START).days
        dates = [DATE_START + timedelta(days=int(d)) for d in self.rng.integers(0, total_days, NUM_RECORDS)]
        dates.sort()

        for i, base_date in enumerate(dates):
            record = self._generate_record(i, base_date)
            self.records.append(record)

            if (i + 1) % 50_000 == 0:
                print(f"  Generated {i + 1:,} / {NUM_RECORDS:,} records")

        df = pd.DataFrame(self.records)
        print(f"Generated {len(df):,} records | Date range: {df['scheduled_departure'].min()} to {df['scheduled_departure'].max()}")
        return df

    def _generate_record(self, idx, base_date):
        """Generate a single shipment record with realistic attributes.

        Args:
            idx: Record index for shipment ID.
            base_date: Base date for scheduling.

        Returns:
            Dict with all shipment fields.
        """
        route = self._pick_route(base_date)
        origin, destination = route
        commodity = self._pick_commodity(route)
        weight_range = COMMODITY_WEIGHTS[commodity]
        weight = round(self.rng.uniform(weight_range[0], weight_range[1]), 1)
        container_count = max(1, int(weight / 25))

        hour = int(self.rng.choice([6, 8, 10, 14, 16, 18, 20, 22]))
        minute = int(self.rng.choice([0, 15, 30, 45]))
        scheduled_departure = base_date.replace(hour=hour, minute=minute)

        base_duration = ROUTE_DURATIONS[route]
        duration_noise = int(self.rng.normal(0, 10))
        travel_minutes = max(base_duration + duration_noise, base_duration // 2)
        scheduled_arrival = scheduled_departure + timedelta(minutes=travel_minutes)

        delay_minutes, delay_cause = self._generate_delay(base_date, commodity)
        actual_departure = scheduled_departure + timedelta(minutes=max(0, delay_minutes // 2))
        actual_arrival = scheduled_arrival + timedelta(minutes=delay_minutes)

        carrier_id = self.rng.choice(CARRIERS)
        customer_id = self.rng.choice(CUSTOMERS)

        return {
            "shipment_id": f"SAR-{base_date.year}{str(idx).zfill(6)}",
            "origin": origin,
            "destination": destination,
            "commodity": commodity,
            "weight_tons": weight,
            "container_count": container_count,
            "scheduled_departure": scheduled_departure,
            "actual_departure": actual_departure,
            "scheduled_arrival": scheduled_arrival,
            "actual_arrival": actual_arrival,
            "delay_minutes": delay_minutes,
            "delay_cause": delay_cause,
            "carrier_id": carrier_id,
            "customer_id": customer_id,
        }

    def _pick_route(self, date):
        """Select a route with seasonal weighting.

        Args:
            date: Shipment date for seasonal adjustment.

        Returns:
            Tuple of (origin, destination).
        """
        weights = np.ones(len(ALL_ROUTES))

        # Mining corridors get extra weight for mineral shipments
        for i, route in enumerate(ALL_ROUTES):
            if route in MINING_CORRIDORS:
                weights[i] = 0.6

        # Hajj season surge on Riyadh-Dammam/Jubail (month varies, approximate)
        month = date.month
        if month in (6, 7):
            for i, route in enumerate(ALL_ROUTES):
                if "Riyadh" in route[0] or "Riyadh" in route[1]:
                    weights[i] *= 1.5

        weights /= weights.sum()
        idx = self.rng.choice(len(ALL_ROUTES), p=weights)
        return ALL_ROUTES[idx]

    def _pick_commodity(self, route):
        """Select commodity type based on route.

        Args:
            route: Tuple of (origin, destination).

        Returns:
            Commodity string.
        """
        if route in MINING_CORRIDORS:
            probs = [0.1, 0.05, 0.55, 0.15, 0.05, 0.1]
        elif "Jubail" in route[0] or "Jubail" in route[1]:
            probs = [0.35, 0.2, 0.1, 0.1, 0.1, 0.15]
        else:
            probs = [0.15, 0.2, 0.1, 0.15, 0.2, 0.2]

        return self.rng.choice(COMMODITIES, p=probs)

    def _generate_delay(self, date, commodity):
        """Generate delay duration and cause with seasonal patterns.

        Args:
            date: Shipment date.
            commodity: Commodity type for delay sensitivity.

        Returns:
            Tuple of (delay_minutes, delay_cause).
        """
        month = date.month

        # Adjust delay probability by season
        on_time_prob = 0.70
        if month in (6, 7, 8):
            on_time_prob -= 0.10  # Summer heat increases delays
        if month in (3, 4):
            on_time_prob -= 0.05  # Ramadan reduced operations
        if month in (6, 7):
            on_time_prob -= 0.05  # Hajj congestion

        roll = self.rng.random()

        if roll < on_time_prob:
            return 0, "none"
        elif roll < on_time_prob + 0.15:
            # Minor delay: 1-4 hours
            minutes = int(self.rng.integers(60, 240))
            cause = self._pick_delay_cause(date, "minor")
            return minutes, cause
        elif roll < on_time_prob + 0.25:
            # Significant delay: 4-24 hours
            minutes = int(self.rng.integers(240, 1440))
            cause = self._pick_delay_cause(date, "significant")
            return minutes, cause
        else:
            # Major delay: 24+ hours
            minutes = int(self.rng.integers(1440, 4320))
            cause = self._pick_delay_cause(date, "major")
            return minutes, cause

    def _pick_delay_cause(self, date, severity):
        """Select delay cause based on season and severity.

        Args:
            date: Shipment date.
            severity: One of 'minor', 'significant', 'major'.

        Returns:
            Delay cause string.
        """
        month = date.month

        if month in (6, 7, 8):
            weights = [0.05, 0.15, 0.15, 0.1, 0.15, 0.25, 0.15, 0.0]
        elif month in (6, 7):
            weights = [0.05, 0.1, 0.3, 0.2, 0.1, 0.1, 0.15, 0.0]
        else:
            weights = [0.1, 0.15, 0.2, 0.15, 0.15, 0.1, 0.15, 0.0]

        causes = DELAY_CAUSES[:-1]  # Exclude "none"
        weights_arr = np.array(weights[:-1])
        weights_arr /= weights_arr.sum()

        if severity == "major":
            # Major delays more likely from maintenance/port issues
            weights_arr[1] *= 1.5
            weights_arr[2] *= 1.5
            weights_arr[5] *= 2.0
            weights_arr /= weights_arr.sum()

        return self.rng.choice(causes, p=weights_arr)

    def save(self, df, path="data/freight_shipments.csv"):
        """Save generated data to CSV.

        Args:
            df: DataFrame to save.
            path: Output file path.

        Returns:
            Path object of saved file.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        print(f"Data saved to {output} ({output.stat().st_size / 1024 / 1024:.1f} MB)")
        return output


def generate_freight_data(output_path="data/freight_shipments.csv"):
    """Convenience function to generate and save freight data.

    Args:
        output_path: Path for output CSV.

    Returns:
        DataFrame with generated records.
    """
    generator = FreightDataGenerator()
    df = generator.generate()
    generator.save(df, output_path)
    return df


if __name__ == "__main__":
    generate_freight_data()

