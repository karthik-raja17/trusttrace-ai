from __future__ import annotations

from typing import Any

from agent.state import AgentState, AuditVerdict
from agent.tools import call_nemotron, call_nemotron_mock


def planner_node(state: AgentState) -> dict[str, Any]:
    """Describe the reasoning plan for the current applicant and print it."""
    applicant_data = state.get("applicant_data") or {}
    shap_values = state.get("shap_values") or {}
    edge_case_type = state.get("edge_case_type") or "none"
    plan = (
        f"Review {len(applicant_data)} applicant fields using "
        f"{len(shap_values)} SHAP values (edge_case_type={edge_case_type}). "
        "Step 1: executor asks Nemotron for a structured, SHAP-grounded rationale. "
        "Step 2: evaluator deterministically checks cited SHAP signs and bias flips."
    )
    print(f"[planner] {plan}")
    return {"plan": plan}


def executor_node(state: AgentState) -> dict[str, Any]:
    """Generate the LLM rationale for the applicant from SHAP-grounded inputs."""
    applicant_data = state.get("applicant_data")
    shap_values = state.get("shap_values")
    if applicant_data is None or shap_values is None:
        raise ValueError(
            "executor_node requires applicant_data and shap_values in state."
        )
    call = call_nemotron_mock if state.get("use_mock") else call_nemotron
    rationale = call(applicant_data, shap_values)
    return {"rationale": rationale}


def evaluator_node(state: AgentState) -> dict[str, Any]:
    """Audit the rationale with deterministic rules and decide approve or rollback."""
    rationale = state.get("rationale")
    if rationale is None:
        raise ValueError("evaluator_node requires a rationale in state.")

    shap_values = state.get("shap_values") or {}
    negative_citations = [
        feature
        for feature in rationale.cited_features
        if shap_values.get(feature, 0.0) < 0
    ]
    rollbacks = state.get("rollbacks", 0)

    if negative_citations:
        action = "rollback"
        violation_type = "hallucination"
        rollbacks += 1
        verdict = AuditVerdict(
            verdict="flag_hallucinated_justification",
            reasons=[
                "The rationale cites features whose SHAP values argue toward default: "
                + ", ".join(negative_citations)
            ],
            violations=negative_citations,
            rollback_required=True,
        )
    elif state.get("edge_case_type") == "bias_flip":
        action = "rollback"
        violation_type = "bias_flip"
        rollbacks += 1
        verdict = AuditVerdict(
            verdict="block_protected_attribute_flip",
            reasons=[
                "The edge case is a bias flip, so the decision must be rolled back."
            ],
            violations=["bias_flip"],
            rollback_required=True,
        )
    else:
        action = "approve"
        violation_type = None
        verdict = AuditVerdict(
            verdict="approve",
            reasons=["No negative SHAP citation or bias flip was detected."],
            violations=[],
            rollback_required=False,
        )

    return {
        "action": action,
        "violation_type": violation_type,
        "verdict": verdict,
        "rollbacks": rollbacks,
    }
