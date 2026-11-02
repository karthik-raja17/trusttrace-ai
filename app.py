from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from agent.graph import MAX_ROLLBACKS, app_graph, state_for_edge_case

load_dotenv()

DATA_PATH = Path(__file__).resolve().parent / "data" / "edge_cases.json"

st.set_page_config(page_title="TrustTrace-AI", layout="wide")


def banner(text: str, background: str, color: str) -> None:
    st.markdown(
        f"""
        <div style="
            background: {background};
            color: {color};
            padding: 1.4rem 1.6rem;
            border-radius: 0.6rem;
            font-size: 1.9rem;
            font-weight: 800;
            letter-spacing: 0.03em;
            text-align: center;
            margin: 1rem 0;
        ">{text}</div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_cases(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["cases"]


def render_planner(update: dict[str, Any]) -> None:
    st.markdown("**Planner**")
    st.info(update.get("plan", "No plan was produced."))


def render_executor(update: dict[str, Any]) -> None:
    st.markdown("**Executor**")
    rationale = update.get("rationale")
    if rationale is None:
        st.warning("The executor did not return a rationale.")
        return
    st.write(rationale.rationale)
    cited = ", ".join(rationale.cited_features) or "none"
    st.caption(
        f"Recommendation: {rationale.recommendation} | Cited SHAP features: {cited}"
    )


def render_evaluator(update: dict[str, Any]) -> None:
    action = update.get("action")
    violation_type = update.get("violation_type")
    rollbacks = update.get("rollbacks", 0)
    st.markdown(f"**Evaluator** | action: `{action}`")
    verdict = update.get("verdict")
    reasons = verdict.reasons if verdict is not None else []
    if action == "rollback":
        st.error("Rollback ordered: " + "; ".join(reasons))
    else:
        st.success("Decision approved: " + "; ".join(reasons))
    st.caption(
        f"Violation type: {violation_type or 'none'} | "
        f"Rollbacks so far: {rollbacks}/{MAX_ROLLBACKS}"
    )


def render_step(node: str, update: dict[str, Any], step: int) -> None:
    with st.container(border=True):
        st.caption(f"Step {step} | {node.title()}")
        if node == "planner":
            render_planner(update)
        elif node == "executor":
            render_executor(update)
        elif node == "evaluator":
            render_evaluator(update)


try:
    cases = load_cases(str(DATA_PATH))
except (OSError, json.JSONDecodeError, KeyError) as exc:
    st.error(f"Could not load edge cases from {DATA_PATH}: {exc}")
    st.stop()

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

with st.sidebar:
    st.header("Case selection")
    labels = [f"{case['id']} - {case['type']}" for case in cases]
    selected_label = st.selectbox("Select an edge case", labels)
    selected_case = cases[labels.index(selected_label)]
    st.markdown(f"**Scenario:** {selected_case['scenario']}")
    st.markdown(
        "**Expected verdict:** "
        f"`{selected_case.get('expected_evaluator_verdict', 'unknown')}`"
    )
    with st.expander("Applicant data"):
        st.json(selected_case["applicant"])
    use_mock = st.toggle("Use Mock LLM (Fast)", value=False)
    if use_mock:
        st.caption("Mock mode: hardcoded rationale, no Nightshift call.")
    run_audit = st.button("Run Audit", type="primary")

st.title("TrustTrace-AI")
st.caption(
    "Autonomous, self-auditing credit underwriting with SHAP-grounded reasoning "
    "and deterministic EU AI Act rollback checks."
)
st.subheader(f"{selected_case['id']}: {selected_case['type'].replace('_', ' ').title()}")

if run_audit:
    st.subheader("Reasoning trace")
    initial_state = state_for_edge_case(selected_case)
    initial_state["use_mock"] = use_mock
    final_state: dict[str, Any] = dict(initial_state)
    saw_rollback = False
    rollback_reasons: list[str] = []
    run_error: str | None = None
    step = 0

    with st.status("Running audit...", expanded=True) as status:
        try:
            for chunk in app_graph.stream(initial_state, stream_mode="updates"):
                for node, update in chunk.items():
                    step += 1
                    final_state.update(update)
                    render_step(node, update, step)
                    if node == "evaluator" and update.get("action") == "rollback":
                        saw_rollback = True
                        verdict = update.get("verdict")
                        if verdict is not None:
                            rollback_reasons.extend(verdict.reasons)
            status.update(
                label=f"Audit complete after {step} reasoning steps",
                state="complete",
            )
        except Exception as exc:
            run_error = str(exc)
            st.error(f"Audit failed: {exc}")
            status.update(label="Audit failed", state="error")

    if run_error is None:
        final_action = final_state.get("action")
        final_rollbacks = int(final_state.get("rollbacks", 0))
        final_verdict = final_state.get("verdict")

        if saw_rollback:
            banner("EU AI ACT VIOLATION DETECTED", "#b00020", "#ffffff")
            st.markdown("**Reason:**")
            for reason in dict.fromkeys(rollback_reasons):
                st.write(f"- {reason}")

        if final_action == "rollback" and final_rollbacks >= MAX_ROLLBACKS:
            banner("ESCALATED TO HUMAN REVIEW", "#f2c94c", "#1f1f1f")

        st.session_state.audit_log.insert(
            0,
            {
                "audit_id": f"ZT-{len(st.session_state.audit_log) + 1:04d}",
                "case_id": selected_case["id"],
                "applicant_id": selected_case.get("source_index", "n/a"),
                "action": final_action or "unknown",
                "violation_type": final_state.get("violation_type") or "none",
                "rollbacks": final_rollbacks,
                "verdict": final_verdict.verdict if final_verdict else "unknown",
                "logged_at": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                ),
            },
        )

st.divider()
st.subheader("Zetaris SQL audit log (mock)")
st.caption(
    "In-memory stand-in for the Zetaris audit table; each completed audit adds one row."
)
if st.session_state.audit_log:
    st.dataframe(st.session_state.audit_log, hide_index=True)
else:
    st.info("No audits recorded yet. Run an audit to populate the log.")

with st.expander("View audit SQL"):
    st.code(
        "SELECT audit_id, case_id, applicant_id, action, violation_type, "
        "rollbacks, verdict, logged_at\n"
        "FROM zetaris.credit_audit_log\n"
        "ORDER BY logged_at DESC;",
        language="sql",
    )
