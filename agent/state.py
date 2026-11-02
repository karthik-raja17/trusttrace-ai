from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

VerdictCode = Literal[
    "approve",
    "reject",
    "block_protected_attribute_flip",
    "flag_hallucinated_justification",
    "deny_then_rollback_and_reapprove",
    "escalate",
]


class RationaleOutput(BaseModel):
    """Structured output returned by Nemotron for a single underwriting decision."""

    rationale: str = Field(
        description=(
            "Plain-language explanation of the credit decision, grounded only in the "
            "supplied applicant data and SHAP values."
        )
    )
    cited_features: list[str] = Field(
        default_factory=list,
        description=(
            "SHAP feature names the rationale explicitly relies on. The deterministic "
            "evaluator checks each citation against shap_values."
        ),
    )
    recommendation: Literal["approve", "reject", "review"] = Field(
        description=(
            "The model's suggested decision. This is informational only; the "
            "evaluator's AuditVerdict is authoritative."
        )
    )


class AuditVerdict(BaseModel):
    """Deterministic audit outcome produced by the rule-based evaluator."""

    verdict: VerdictCode = Field(
        description=(
            "Final audit action: approve, reject, block a protected-attribute flip, "
            "flag a hallucinated justification, deny then roll back and re-approve, "
            "or escalate for human review."
        )
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Human-readable reasons supporting the verdict, in evaluation order.",
    )
    violations: list[str] = Field(
        default_factory=list,
        description=(
            "Specific rule identifiers or descriptions that were violated, such as a "
            "protected attribute driving the decision or a SHAP contradiction."
        ),
    )
    rollback_required: bool = Field(
        default=False,
        description=(
            "True when the agent must discard the prior decision and re-run the "
            "executor with corrected inputs or reasoning."
        ),
    )


class AgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes.

    Attributes:
        applicant_data: Raw applicant features used for scoring and reasoning.
        shap_values: Per-feature SHAP contributions in log-odds for the applicant.
        rationale: Structured Nemotron rationale and SHAP citations.
        verdict: Deterministic evaluator verdict for the decision.
        plan: Human-readable plan printed and returned by the planner node.
        edge_case_type: Adversarial case category, such as bias_flip or contradiction.
        action: Control-flow decision emitted by the evaluator.
        violation_type: Reason the evaluator ordered a rollback, if any.
        rollbacks: Number of rollback verdicts emitted so far, used to cap the loop.
        use_mock: When true, the executor uses the deterministic mock LLM.
    """

    applicant_data: dict[str, Any]
    shap_values: dict[str, float]
    rationale: RationaleOutput | None
    verdict: AuditVerdict | None
    plan: str | None
    edge_case_type: str | None
    action: Literal["approve", "rollback"] | None
    violation_type: Literal["hallucination", "bias_flip"] | None
    rollbacks: int
    use_mock: bool
