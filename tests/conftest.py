"""
Pytest configuration and shared fixtures.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def synthetic_dataset():
    """Generate a synthetic dataset with mixed types and issues."""
    np.random.seed(42)
    n = 200

    data = {
        # Numeric continuous
        "age": np.random.normal(40, 15, n),
        "income": np.random.exponential(50000, n),
        # Numeric discrete
        "num_children": np.random.randint(0, 5, n),
        "purchases": np.random.randint(0, 100, n),
        # Categorical
        "gender": np.random.choice(["M", "F", "Otro"], n, p=[0.45, 0.45, 0.1]),
        "region": np.random.choice(["Nord", "Centro", "Sud"], n),
        "category": np.random.choice(["A", "B", "C", "D"], n),
        # Boolean
        "premium": np.random.choice([True, False], n),
        # Text
        "comment": [f"Comment {i}" for i in range(n)],
        # Identifier-like
        "user_id": [f"USR{i:06d}" for i in range(n)],
        # Datetime
        "signup_date": pd.date_range("2020-01-01", periods=n, freq="D"),
        # With nulls
        "nullable_col": np.where(np.random.random(n) > 0.2, np.random.randint(1, 100, n), np.nan),
    }

    df = pd.DataFrame(data)

    # Add constant column
    df["constant"] = "X"

    # Add quasi-constant column
    df["quasi_constant"] = np.where(np.random.random(len(df)) > 0.01, "A", "B")

    # Add high-cardinality column
    df["high_cardinality"] = [f"VAL{i}" for i in range(len(df))]

    # Add outliers
    outlier_indices = np.random.choice(len(df), 5, replace=False)
    df.loc[outlier_indices, "age"] = 150

    # Add exact duplicate rows. This must happen LAST, after every column
    # (including the per-row-unique `high_cardinality`) has its final value,
    # otherwise the "duplicated" rows would differ from their originals and
    # no exact duplicates would actually exist.
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    return df


@pytest.fixture
def classification_dataset():
    """Generate a classification dataset."""
    np.random.seed(42)
    n = 300

    df = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, n),
            "feature2": np.random.normal(0, 1, n),
            "feature3": np.random.choice(["A", "B", "C"], n),
            "target": np.random.choice([0, 1], n, p=[0.7, 0.3]),  # Imbalanced
        }
    )
    return df


@pytest.fixture
def regression_dataset():
    """Generate a regression dataset."""
    np.random.seed(42)
    n = 300

    x = np.random.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "feature1": x,
            "feature2": np.random.normal(0, 1, n),
            "feature3": np.random.choice(["A", "B", "C"], n),
            "target": x * 2 + np.random.normal(0, 0.5, n),
        }
    )
    return df


@pytest.fixture
def csv_file(tmp_output_dir, synthetic_dataset):
    """Save synthetic dataset as CSV."""
    csv_path = tmp_output_dir / "data.csv"
    synthetic_dataset.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def latin1_csv_file(tmp_output_dir):
    """Create a CSV with latin-1 encoding and special characters."""
    csv_path = tmp_output_dir / "data_latin1.csv"
    df = pd.DataFrame(
        {
            "nombre": ["José", "María", "Ramón"],
            "región": ["México", "España", "Argentina"],
            "valor": [100.5, 200.3, 150.7],
        }
    )
    df.to_csv(csv_path, index=False, encoding="latin-1", sep=";", decimal=",")
    return csv_path


@pytest.fixture
def late_accent_cp1252_file(tmp_output_dir):
    """
    A cp1252 CSV where the first non-ASCII byte appears well past the old
    10 KB chardet sample window, used to regression-test detect_encoding.
    """
    path = tmp_output_dir / "late_accent.csv"
    header = "id;nombre;valor\n"
    padding = "".join(f"{i};Nombre Generico {i};{i}.0\n" for i in range(1, 700))
    accented_row = "99999;José Muñoz Peña;2.0\n"
    content = header + padding + accented_row
    raw = content.encode("cp1252")
    assert len(raw) > 10_000, "fixture must exceed the old 10 KB sample window"
    path.write_bytes(raw)
    return path


@pytest.fixture
def freetext_spaces_csv_file(tmp_output_dir):
    """CSV whose true delimiter is ',' but a free-text column contains many spaces."""
    path = tmp_output_dir / "freetext.csv"
    rows = ["id,comment,score"]
    for i in range(20):
        rows.append(f'{i},"This is a long comment with many spaces in it {i}",{i * 1.5}')
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


@pytest.fixture
def semicolon_decimal_comma_csv_file(tmp_output_dir):
    """CSV delimited by ';' where numeric fields use ',' as the decimal separator."""
    path = tmp_output_dir / "semicolon_decimals.csv"
    rows = ["id;amount;rating;flag"]
    for i in range(20):
        rows.append(f"{i};{i},5;{i % 5},0;True")
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


@pytest.fixture
def multi_sheet_excel_file(tmp_output_dir):
    """Excel workbook with two sheets, to test that the default load picks the first one."""
    path = tmp_output_dir / "multi_sheet.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"a": [1, 2, 3]}).to_excel(writer, sheet_name="Primera", index=False)
        pd.DataFrame({"b": [4, 5, 6]}).to_excel(writer, sheet_name="Segunda", index=False)
    return path


@pytest.fixture
def json_array_file(tmp_output_dir):
    """Single-line JSON array of records with a nested dict and a nested list."""
    path = tmp_output_dir / "records.json"
    records = [
        {"id": 1, "name": "A", "address": {"city": "X", "zip": "111"}, "tags": ["a", "b"]},
        {"id": 2, "name": "B", "address": {"city": "Y", "zip": "222"}, "tags": ["c"]},
        {"id": 3, "name": "C", "address": {"city": "Z", "zip": "333"}, "tags": []},
    ]
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


@pytest.fixture
def jsonl_file(tmp_output_dir):
    """JSON Lines file (one record per line)."""
    path = tmp_output_dir / "records.jsonl"
    records = [{"id": i, "value": i * 2} for i in range(5)]
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path
