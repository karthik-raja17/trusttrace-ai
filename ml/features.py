from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "german_credit.csv"
MODEL_DIR = PROJECT_ROOT / "ml"
MODEL_PATH = MODEL_DIR / "model.json"
FEATURES_PATH = MODEL_DIR / "features.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET_COLUMN = "default"
BAD_VALUE = 2
DECISION_THRESHOLD = 0.5

NUMERIC_COLUMNS = [
    "months_loan_duration",
    "amount",
    "installment_rate",
    "residence_history",
    "age",
    "existing_credits",
    "dependents",
]

CATEGORICAL_COLUMNS = [
    "checking_balance",
    "credit_history",
    "purpose",
    "savings_balance",
    "employment_length",
    "marital_status",
    "other_debtors",
    "property",
    "installment_plan",
    "housing",
    "job",
    "sex",
    "telephone",
    "foreign_worker",
]

PROTECTED_COLUMNS = ["sex", "age"]


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_").lower()
    return cleaned or "unknown"


def derive_protected_columns(frame: pd.DataFrame) -> pd.DataFrame:
    status = frame["personal_status"].astype(str).str.strip().str.lower()
    frame = frame.copy()
    frame["sex"] = np.where(status.str.contains("female"), "female", "male")
    frame["marital_status"] = np.select(
        [
            status.str.contains("married"),
            status.str.contains("divorced"),
            status.str.contains("single"),
        ],
        ["married", "divorced", "single"],
        default="unknown",
    )
    return frame


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"German credit CSV not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path.name} has no rows")
    required = set(NUMERIC_COLUMNS) | {"personal_status"} | {
        column for column in CATEGORICAL_COLUMNS if column not in {"sex", "marital_status"}
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing column(s): {', '.join(missing)}")
    return derive_protected_columns(frame)


def make_target(frame: pd.DataFrame) -> np.ndarray:
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"{TARGET_COLUMN} column is required to train the model")
    return (frame[TARGET_COLUMN].astype(int) == BAD_VALUE).astype(int).to_numpy()


def fit_feature_spec(frame: pd.DataFrame) -> dict:
    spec = {
        "numeric_columns": list(NUMERIC_COLUMNS),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "categories": {},
        "category_features": {},
        "numeric_medians": {},
        "feature_names": [],
    }
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        spec["numeric_medians"][column] = float(values.median())
    for column in CATEGORICAL_COLUMNS:
        spec["categories"][column] = sorted(frame[column].astype(str).unique().tolist())

    names = list(NUMERIC_COLUMNS)
    seen = set(names)
    for column in CATEGORICAL_COLUMNS:
        spec["category_features"][column] = {}
        for category in spec["categories"][column]:
            base = safe_name(f"{column}__{category}")
            name = base
            counter = 2
            while name in seen:
                name = f"{base}_{counter}"
                counter += 1
            seen.add(name)
            names.append(name)
            spec["category_features"][column][category] = name
    spec["feature_names"] = names
    return spec


def transform(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    rows = len(frame)
    data: dict[str, np.ndarray] = {}
    for column in spec["numeric_columns"]:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        data[column] = np.where(np.isfinite(values), values, spec["numeric_medians"][column])
    for column in spec["categorical_columns"]:
        values = frame[column].astype(str).to_numpy()
        for category, name in spec["category_features"][column].items():
            data[name] = (values == category).astype(float)
    for name in spec["feature_names"]:
        if name not in data:
            data[name] = np.zeros(rows, dtype=float)
    return pd.DataFrame(data, index=frame.index)[spec["feature_names"]]


def save_feature_spec(spec: dict, path: Path = FEATURES_PATH) -> None:
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")


def load_feature_spec(path: Path = FEATURES_PATH) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `python ml/train.py` before generating edge cases."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def describe_row(row: pd.Series) -> dict:
    return {
        column: (int(row[column]) if column in NUMERIC_COLUMNS else str(row[column]))
        for column in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    }
