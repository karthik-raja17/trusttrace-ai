from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.nodes import evaluator_node, executor_node, planner_node
from agent.state import AgentState

MAX_ROLLBACKS = 2


def route_after_evaluator(
    state: AgentState,
) -> Literal["rollback", "approve", "stop"]:
    """Send rollback verdicts back to the planner until the retry cap is hit."""
    if state.get("action") == "rollback":
        if state.get("rollbacks", 0) < MAX_ROLLBACKS:
            return "rollback"
        return "stop"
    return "approve"


def build_app_graph():
    """Wire planner, executor, and evaluator into the audit loop."""
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("evaluator", evaluator_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {"rollback": "planner", "approve": END, "stop": END},
    )
    return graph.compile()


app_graph = build_app_graph()


def load_first_edge_case() -> dict[str, Any]:
    """Load the first adversarial case from the generated edge-case file."""
    path = PROJECT_ROOT / "data" / "edge_cases.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["cases"][0]


def state_for_edge_case(case: dict[str, Any]) -> AgentState:
    """Convert a serialized edge case into the graph's initial state."""
    stored_shap = case.get("shap_values")
    if stored_shap:
        shap_values = {
            str(name): float(value) for name, value in stored_shap.items()
        }
    else:
        shap_values = {
            item["feature"]: float(item["contribution"])
            for item in case.get("top_shap", [])
        }
    return {
        "applicant_data": case["applicant"],
        "shap_values": shap_values,
        "edge_case_type": case["type"],
        "rollbacks": 0,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    case = load_first_edge_case()
    print(f"[graph] Running edge case {case['id']} ({case['type']})")
    result = app_graph.invoke(state_for_edge_case(case))
    rationale = result.get("rationale")
    verdict = result.get("verdict")
    print("action:", result.get("action"))
    print("violation_type:", result.get("violation_type"))
    print("rollbacks:", result.get("rollbacks"))
    if verdict is not None:
        print("verdict:", verdict.model_dump())
    if rationale is not None:
        print("rationale:", rationale.rationale)
