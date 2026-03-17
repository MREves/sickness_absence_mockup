import polars as pl
import numpy as np
from datetime import datetime, timedelta

def generate_nhs_dummy_data(n_staff=1000):
    # 1. Base Staff Data
    roles_map = {
        'Nurse': 'Clinical', 
        'Doctor': 'Clinical', 
        'AHW': 'Clinical', 
        'Admin': 'Non-Clinical', 
        'Manager': 'Non-Clinical'
    }

    np.random.seed(42)  # For reproducibility
    
    # Generate base features
    staff_df = pl.DataFrame({
        'staff_id': np.arange(n_staff),
        'age': np.random.randint(22, 66, n_staff),
        'gender': np.random.choice(['M', 'F'], n_staff),
        'role': np.random.choice(list(roles_map.keys()), n_staff),
        'imd_quintile': np.random.choice([1, 2, 3, 4, 5], n_staff, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'tenure_years': np.random.uniform(0, 25, n_staff),
        'prev_absences': np.random.poisson(1.5, n_staff)
    })

    # Map clinical status
    staff_df = staff_df.with_columns(
        pl.col("role").replace(roles_map).alias("is_clinical")
    )

    # 2. Vectorized Hazard/Duration Logic
    # In Polars, we calculate the 'risk' for everyone at once (vectorized)
    staff_df = staff_df.with_columns(
        risk_score = (
            5.0 + 
            pl.when(pl.col("imd_quintile") == 1).then(4.0).otherwise(0.0) +
            pl.when(pl.col("is_clinical") == "Clinical").then(3.0).otherwise(0.0) +
            pl.when(pl.col("age") >= 45).then(3.0).otherwise(0.0) +
            pl.when((pl.col("gender") == "F") & (pl.col("age") >= 40)).then(5.0).otherwise(0.0)
        )
    )

    # Generate durations using the risk_score as the scale
    # Polars allows us to map a numpy function across a column
    durations = np.random.exponential(scale=staff_df["risk_score"].to_numpy()) + 1
    
    staff_df = staff_df.with_columns(
        duration_days = pl.Series(durations).cast(pl.Int64),
        # 1 = returned to work (event), 0 = censored
        event = pl.when(pl.Series(durations) > 45).then(
            pl.Series(np.random.choice([0, 1], n_staff, p=[0.4, 0.6]))
        ).otherwise(1)
    ).drop("risk_score")

    # 3. Time Series Data
    dates = pl.date_range(start=datetime(2022, 1, 1), end=datetime(2025, 12, 31), interval="1d", eager=True)
    ts_df = pl.DataFrame({
        "date": dates,
        "y": [int(20 + 8*np.sin(i/30) + np.random.normal(0, 3)) for i in range(len(dates))]
    })

    print(f"Synthesized {n_staff} records using Polars.")
    return ts_df, staff_df

# Execute
#ts_df, staff_df = generate_nhs_dummy_data()