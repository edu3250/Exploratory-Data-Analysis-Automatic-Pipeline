"""
Generate synthetic datasets for testing and demonstration.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_ecommerce_dataset(n_rows: int = 500) -> pd.DataFrame:
    """Generate a synthetic e-commerce dataset."""
    np.random.seed(42)

    dates = pd.date_range("2023-01-01", periods=n_rows, freq="D")

    data = {
        "date": dates,
        "customer_id": [f"CUST{i:06d}" for i in range(n_rows)],
        "age": np.random.normal(42, 15, n_rows).astype(int).clip(18, 80),
        "gender": np.random.choice(["M", "F", "Otro"], n_rows, p=[0.48, 0.48, 0.04]),
        "region": np.random.choice(["Norte", "Centro", "Sur", "Este", "Oeste"], n_rows),
        "product_category": np.random.choice(["Electrónica", "Ropa", "Hogar", "Deporte", "Libros"], n_rows),
        "amount_spent": np.random.exponential(150, n_rows).clip(10, 1000),
        "items_purchased": np.random.randint(1, 20, n_rows),
        "is_premium": np.random.choice([True, False], n_rows, p=[0.15, 0.85]),
        "previous_purchases": np.random.randint(0, 50, n_rows),
        "review_rating": np.random.choice([1, 2, 3, 4, 5], n_rows, p=[0.05, 0.1, 0.15, 0.25, 0.45]),
    }

    df = pd.DataFrame(data)

    # Add some nulls
    null_cols = ["review_rating"]
    for col in null_cols:
        null_indices = np.random.choice(len(df), size=int(0.1 * len(df)), replace=False)
        df.loc[null_indices, col] = np.nan

    # Add target: is_customer_churn
    df["is_churn"] = ((df["previous_purchases"] < 5) & (df["amount_spent"] < 100) | (df["review_rating"] <= 2)).astype(
        int
    )
    df.loc[np.random.choice(len(df), size=20, replace=False), "is_churn"] = (
        1 - df.loc[np.random.choice(len(df), size=20, replace=False), "is_churn"]
    )

    # Add duplicates
    df = pd.concat([df, df.iloc[:20]], ignore_index=True)

    return df.sample(frac=1).reset_index(drop=True)


def generate_healthcare_dataset(n_rows: int = 400) -> pd.DataFrame:
    """Generate a synthetic healthcare dataset."""
    np.random.seed(43)

    data = {
        "patient_id": [f"PAT{i:05d}" for i in range(n_rows)],
        "age": np.random.normal(55, 20, n_rows).astype(int).clip(18, 95),
        "weight_kg": np.random.normal(75, 15, n_rows),
        "height_cm": np.random.normal(170, 10, n_rows),
        "systolic_bp": np.random.normal(120, 15, n_rows),
        "diastolic_bp": np.random.normal(80, 10, n_rows),
        "cholesterol": np.random.normal(200, 50, n_rows),
        "glucose": np.random.normal(100, 30, n_rows),
        "gender": np.random.choice(["M", "F"], n_rows, p=[0.45, 0.55]),
        "smoking": np.random.choice(["No", "Anterior", "Actual"], n_rows, p=[0.6, 0.25, 0.15]),
        "exercise_freq": np.random.choice(["Nunca", "Raramente", "A veces", "Frecuente", "Muy Frecuente"], n_rows),
        "visit_date": pd.date_range("2023-01-01", periods=n_rows, freq="h"),
        "notes": [f"Visit notes {i}" for i in range(n_rows)],
    }

    df = pd.DataFrame(data)

    # Add target: has_disease
    df["has_disease"] = (
        (df["age"] > 60) & (df["cholesterol"] > 250) | (df["glucose"] > 150) | (df["systolic_bp"] > 140)
    ).astype(int)

    # Add nulls
    null_idx = np.random.choice(len(df), size=int(0.08 * len(df)), replace=False)
    df.loc[null_idx, "glucose"] = np.nan

    return df


def generate_sales_dataset(n_rows: int = 1000) -> pd.DataFrame:
    """Generate a synthetic sales dataset for regression."""
    np.random.seed(44)

    data = {
        "date": pd.date_range("2022-01-01", periods=n_rows, freq="D"),
        "region": np.random.choice(["NOAM", "EMEA", "APAC", "LATAM"], n_rows),
        "sales_rep": [f"REP{i % 50:02d}" for i in range(n_rows)],
        "product_line": np.random.choice(["Product A", "Product B", "Product C", "Product D"], n_rows),
        "units_sold": np.random.randint(1, 100, n_rows),
        "unit_price": np.random.normal(50, 20, n_rows).clip(10, 200),
        "customer_satisfaction": np.random.uniform(1, 5, n_rows),
        "marketing_spend": np.random.exponential(500, n_rows),
        "days_to_close": np.random.randint(1, 60, n_rows),
    }

    df = pd.DataFrame(data)

    # Add regression target: total_revenue
    df["total_revenue"] = (
        df["units_sold"] * df["unit_price"] + df["marketing_spend"] * 0.5 + np.random.normal(0, 1000, n_rows)
    ).clip(0, None)

    # Add some nulls
    null_idx = np.random.choice(len(df), size=int(0.05 * len(df)), replace=False)
    df.loc[null_idx, "customer_satisfaction"] = np.nan

    return df


def main():
    """Generate all sample datasets."""
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Generando datasets sinteticos...")

    # E-commerce dataset
    df_ecom = generate_ecommerce_dataset()
    csv_path = data_dir / "ecommerce.csv"
    df_ecom.to_csv(csv_path, index=False)
    print(f"[OK] {csv_path} ({len(df_ecom)} rows)")

    # Healthcare dataset
    df_health = generate_healthcare_dataset()
    csv_path = data_dir / "healthcare.csv"
    df_health.to_csv(csv_path, index=False)
    print(f"[OK] {csv_path} ({len(df_health)} rows)")

    # Sales dataset
    df_sales = generate_sales_dataset()
    csv_path = data_dir / "sales.csv"
    df_sales.to_csv(csv_path, index=False)
    print(f"[OK] {csv_path} ({len(df_sales)} rows)")

    # Also save as Excel
    xlsx_path = data_dir / "ecommerce.xlsx"
    df_ecom.to_excel(xlsx_path, index=False, sheet_name="Data")
    print(f"[OK] {xlsx_path}")

    # Save as Parquet
    parquet_path = data_dir / "sales.parquet"
    df_sales.to_parquet(parquet_path, index=False)
    print(f"[OK] {parquet_path}")

    # Save as JSON
    json_path = data_dir / "healthcare.json"
    df_health.to_json(json_path, orient="records", date_format="iso")
    print(f"[OK] {json_path}")

    # Save with special encoding (latin-1) and delimiter (;)
    latin1_path = data_dir / "data_latin1.csv"
    df_ecom_sample = df_ecom.head(50).copy()
    df_ecom_sample.to_csv(latin1_path, index=False, encoding="latin-1", sep=";", decimal=",")
    print(f"[OK] {latin1_path} (latin-1, semicolon, decimal comma)")

    print("\n[SUCCESS] Todos los datasets generados en data/raw/")


if __name__ == "__main__":
    main()
