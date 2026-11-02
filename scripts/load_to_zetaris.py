import base64
import json
import math
import os
import re
import uuid
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_ROW_COUNT = 1000
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_LOCAL_STORAGE_ROOT = "/home/zetaris/customer-data/shared"


def find_csv() -> Path:
    candidates = (
        PROJECT_ROOT / "german_credit.csv",
        PROJECT_ROOT / "data" / "german_credit.csv",
        PROJECT_ROOT / "data" / "credit.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = PROJECT_ROOT / "german_credit.csv"
    raise FileNotFoundError(f"Could not find the German credit CSV. Expected {expected}")


def quote_identifier(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def sql_literal(value, dtype) -> str:
    if pd.isna(value):
        return "NULL"
    if pd.api.types.is_bool_dtype(dtype):
        return "TRUE" if bool(value) else "FALSE"
    if pd.api.types.is_integer_dtype(dtype):
        return str(int(value))
    if pd.api.types.is_float_dtype(dtype):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Cannot load non-finite numeric value: {value}")
        return repr(number)
    text = str(value)
    if any(character in text for character in ('"', "\r", "\n")):
        raise ValueError(
            "Values containing double quotes or line breaks are not supported by "
            f"the Zetaris export statement: {text!r}"
        )
    return "'" + text.replace("'", "''") + "'"


def build_values_query(dataframe: pd.DataFrame) -> str:
    rows = []
    for record in dataframe.itertuples(index=False, name=None):
        literals = ", ".join(
            sql_literal(value, dataframe.dtypes.iloc[index])
            for index, value in enumerate(record)
        )
        rows.append(f"({literals})")
    alias = ", ".join(quote_identifier(str(column)) for column in dataframe.columns)
    return f"SELECT * FROM VALUES {', '.join(rows)} AS t({alias})"


def run_sql(
    session: requests.Session,
    url: str,
    org_id: str,
    api_key: str,
    sql: str,
    limit: int = 200,
) -> dict:
    response = session.post(
        url.rstrip("/") + "/sql-editor/sqls/run",
        headers={
            "Authorization": f"Bearer {api_key}",
            "X-Org-ID": org_id,
            "X-Request-ID": str(uuid.uuid4()),
        },
        json={
            "queryId": str(uuid.uuid4()),
            "sql": sql,
            "source": "SqlEditor",
            "limit": limit,
        },
        timeout=300,
    )
    try:
        result = response.json()
    except ValueError:
        result = None
    if not response.ok:
        message = result.get("msg") if isinstance(result, dict) else response.text
        raise RuntimeError(f"Zetaris rejected SQL ({response.status_code}): {message}")
    if isinstance(result, dict) and (result.get("success") is False or result.get("error")):
        message = result.get("error") or result.get("message") or "unknown error"
        raise RuntimeError(f"Zetaris rejected SQL: {message}")
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected Zetaris response: {response.text[:500]}")
    return result


def list_names(result: dict, column: int = 0) -> set[str]:
    rows = result.get("data") or []
    return {str(row[column]) for row in rows if len(row) > column}


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    rest_url = os.environ.get("ZETARIS_REST_URL")
    org_id = os.environ.get("ZETARIS_ORG_ID")
    api_key = os.environ.get("ZETARIS_API_KEY")
    missing = [
        name
        for name, value in (
            ("ZETARIS_REST_URL", rest_url),
            ("ZETARIS_ORG_ID", org_id),
            ("ZETARIS_API_KEY", api_key),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required .env setting(s): {', '.join(missing)}")

    database = os.environ.get("ZETARIS_DATABASE", "german_credit_db")
    table_name = os.environ.get("ZETARIS_TABLE_NAME", "german_credit")
    for label, identifier in (("ZETARIS_DATABASE", database), ("ZETARIS_TABLE_NAME", table_name)):
        if not IDENTIFIER_PATTERN.fullmatch(identifier):
            raise ValueError(f"{label} must be a simple SQL identifier")

    local_root = os.environ.get(
        "ZETARIS_LOCAL_STORAGE_ROOT", DEFAULT_LOCAL_STORAGE_ROOT
    ).rstrip("/")
    export_path = os.environ.get(
        "ZETARIS_EXPORT_PATH", f"{local_root}/{database}_{table_name}"
    )
    if not export_path.startswith("/") or '"' in export_path:
        raise ValueError(
            "ZETARIS_EXPORT_PATH must be an absolute path inside the Zetaris local "
            "storage root and must not contain double quotes"
        )

    csv_path = find_csv()
    dataframe = pd.read_csv(csv_path)
    if len(dataframe) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ROW_COUNT} rows in {csv_path.name}, found {len(dataframe)}"
        )
    if dataframe.empty or not len(dataframe.columns):
        raise ValueError(f"{csv_path.name} has no data columns")
    if not dataframe.columns.is_unique:
        raise ValueError(f"{csv_path.name} contains duplicate column names")
    invalid_columns = [
        str(column)
        for column in dataframe.columns
        if not IDENTIFIER_PATTERN.fullmatch(str(column))
    ]
    if invalid_columns:
        raise ValueError(f"CSV column names must be simple SQL identifiers: {invalid_columns}")

    query = build_values_query(dataframe)
    options_base64 = base64.b64encode(json.dumps({}).encode()).decode()
    export_sql = (
        'RUN EXPORT TO STORAGE (query = "'
        + query
        + '", path = "'
        + export_path
        + '", format = "parquet", saveMode = "overwrite", optionsBase64 = "'
        + options_base64
        + '")'
    )

    with requests.Session() as session:
        databases = list_names(
            run_sql(session, rest_url, org_id, api_key, "SHOW LIGHTNING DATABASES", limit=500)
        )
        if database not in databases:
            run_sql(
                session,
                rest_url,
                org_id,
                api_key,
                f"CREATE LIGHTNING DATABASE {database} "
                "DESCRIBE BY 'German credit data loaded from german_credit.csv'",
            )

        tables = list_names(
            run_sql(
                session,
                rest_url,
                org_id,
                api_key,
                f"SHOW LIGHTNING TABLES {database}",
                limit=500,
            )
        )
        if table_name in tables:
            run_sql(
                session,
                rest_url,
                org_id,
                api_key,
                f"DROP LIGHTNING DATABASE TABLE {database}.{table_name}",
            )

        export_result = run_sql(
            session, rest_url, org_id, api_key, export_sql, limit=1
        )
        rows = export_result.get("data") or []
        if not rows or str(rows[0][1]).upper() != "SUCCESS":
            raise RuntimeError(f"Zetaris export did not succeed: {export_result}")

        run_sql(
            session,
            rest_url,
            org_id,
            api_key,
            f"CREATE LIGHTNING FILESTORE TABLE {table_name} FROM {database} "
            f'FORMAT PARQUET OPTIONS (path "{export_path}")',
        )

        quoted_table = f"{quote_identifier(database)}.{quote_identifier(table_name)}"
        count_result = run_sql(
            session, rest_url, org_id, api_key, f"SELECT COUNT(*) FROM {quoted_table}"
        )
        data = count_result.get("data") or []
        if not data or not data[0]:
            raise RuntimeError("Zetaris did not return a row count for verification")
        actual_count = int(data[0][0])
        if actual_count != EXPECTED_ROW_COUNT:
            raise RuntimeError(
                f"Row count verification failed: expected {EXPECTED_ROW_COUNT}, "
                f"found {actual_count}"
            )

        if "amount" in dataframe.columns:
            sum_result = run_sql(
                session,
                rest_url,
                org_id,
                api_key,
                f"SELECT SUM(amount) FROM {quoted_table}",
            )
            sum_rows = sum_result.get("data") or []
            if not sum_rows or not sum_rows[0] or sum_rows[0][0] is None:
                raise RuntimeError("Zetaris did not return an amount total for verification")
            actual_sum = int(round(float(sum_rows[0][0])))
            expected_sum = int(dataframe["amount"].sum())
            if actual_sum != expected_sum:
                raise RuntimeError(
                    f"Amount total verification failed: expected {expected_sum}, "
                    f"found {actual_sum}"
                )

    print(
        f"Successfully loaded {actual_count} rows from {csv_path.name} into "
        f"{database}.{table_name}."
    )


if __name__ == "__main__":
    main()
