from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from agent.state import RationaleOutput

DEFAULT_MODEL = "meterless/auto"

SYSTEM_PROMPT = (
    "You are a credit underwriting analyst explaining a machine-learning decision. "
    "Ground every statement in the supplied applicant data and SHAP values. "
    "Only cite features that appear in the SHAP values, and never claim a feature "
    "supports approval when its SHAP contribution points toward default. "
    "Return a concise rationale, the cited SHAP feature names, and a recommendation."
)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"{name} is not set. Add it to .env before calling the model."
        )
    return value


def call_nemotron(
    applicant_data: dict[str, Any], shap_values: dict[str, float]
) -> RationaleOutput:
    """Request a structured, SHAP-grounded rationale from Nemotron via Meterless."""
    client = OpenAI(
        base_url=_require_env("MODEL_BASE_URL"),
        api_key=_require_env("MODEL_API_KEY"),
    )
    payload = json.dumps(
        {"applicant_data": applicant_data, "shap_values": shap_values},
        indent=2,
        default=str,
    )
    response = client.beta.chat.completions.parse(
        model=os.getenv("NEMOTRON_MODEL", DEFAULT_MODEL),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
        response_format=RationaleOutput,
    )
    if not response.choices:
        raise RuntimeError("Nemotron returned no choices.")
    parsed = response.choices[0].message.parsed
    if parsed is None:
        refusal = response.choices[0].message.refusal
        raise RuntimeError(f"Nemotron did not return a parsed rationale: {refusal}")
    return parsed


def call_nemotron_mock(
    applicant_data: dict[str, Any], shap_values: dict[str, float]
) -> RationaleOutput:
    """Return a deterministic rationale so the graph can run without Nightshift."""
    return RationaleOutput(
        rationale="Approved due to stable employment",
        cited_features=["employment_since"],
        recommendation="approve",
    )


def log_to_zetaris(applicant_id: str | int, reason: str) -> None:
    """Stub for writing an audit row to Zetaris; prints until the SQL client lands."""
    print(f"[zetaris] applicant={applicant_id} reason={reason}")
