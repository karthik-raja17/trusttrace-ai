# TrustTrace-AI

An autonomous, self-auditing credit underwriting engine that pairs a quantitative
ML model with qualitative LLM reasoning, then audits the result with deterministic
rules before a decision is allowed to stand.

Built for Hackathon Track 3 (Reasoning Architecture): the system must detect
algorithmic bias, ground every explanation in SHAP values, and roll back unsafe
decisions for EU AI Act compliance.

## Why

Credit models are opaque, and LLM-generated justifications can confidently cite
the wrong evidence. TrustTrace-AI makes the reasoning trace auditable:

- Every decision is explained using the model's own SHAP contributions.
- A **deterministic** evaluator (plain Python, not another LLM) checks the
  explanation against the SHAP signs and against protected-attribute bias flips.
- Unsafe reasoning triggers a rollback and re-plan, up to a retry cap, then
  escalates to a human.
- Every completed audit is written to an audit log with a timestamped verdict.

## Architecture

```text
                 ┌─────────────┐
   edge case ───▶│   Planner   │  describes the audit plan
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  Executor   │  Nightshift / Nemotron produces a structured,
                 └──────┬──────┘  SHAP-grounded rationale (Pydantic schema)
                        ▼
                 ┌─────────────┐
                 │  Evaluator  │  deterministic checks:
                 └──────┬──────┘  • cited feature has negative SHAP?
                        │         • protected-attribute bias flip?
             ┌──────────┴──────────┐
             ▼                     ▼
        rollback loop          approve / stop
     (capped at 2 retries)   (escalate to human)
```

The loop is implemented as a LangGraph state machine in `agent/graph.py`.

## Features

- **Adversarial edge cases** generated from the real dataset: bias flips,
  SHAP contradictions, and high-risk rollbacks (`ml/edge_cases.py`).
- **Grounded rationales**: the executor must return a schema with the rationale,
  cited SHAP features, and a recommendation.
- **Deterministic guardrails**: the evaluator compares citations against SHAP
  signs in Python; the LLM never grades itself.
- **Rollback + escalation**: unsafe verdicts re-enter the planner until
  `MAX_ROLLBACKS` is reached.
- **Live UI**: Streamlit renders the reasoning trace step by step, violation
  banners, rollback count, and an in-memory Zetaris-style audit log.
- **Mock mode**: run the whole graph without any external model call.

## Tech stack

| Layer | Technology |
| --- | --- |
| Agent orchestration | LangGraph, Pydantic |
| ML | XGBoost, scikit-learn, SHAP |
| Data / audit | Zetaris (REST SQL API) |
| LLM | NVIDIA Nemotron via Meterless Nightshift |
| UI | Streamlit |

## Repository layout

```text
.
├── app.py                  # Streamlit UI
├── agent/
│   ├── state.py            # AgentState + Pydantic output schemas
│   ├── nodes.py            # planner / executor / evaluator nodes
│   ├── tools.py            # Nightshift call + mock
│   └── graph.py            # LangGraph wiring + CLI demo
├── ml/
│   ├── features.py         # feature spec and transforms
│   ├── train.py            # trains XGBoost, writes model/metrics
│   ├── edge_cases.py       # generates data/edge_cases.json
│   └── model.json          # trained model (generated)
├── data/
│   ├── german_credit.csv   # UCI German Credit baseline
│   └── edge_cases.json     # generated adversarial cases
├── scripts/
│   ├── load_to_zetaris.py  # loads the CSV into Zetaris via REST
│   └── smoke_test.py       # checks Zetaris + Nemotron connectivity
└── requirements.txt
```

## Quickstart

```bash
git clone <your-repo-url>
cd open-agent-hack

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env        # then fill in real credentials
```

### 1. Train the model

```bash
python ml/train.py
```

Writes `ml/model.json`, `ml/features.json`, and `ml/metrics.json`
(5-fold CV AUC ≈ 0.79, holdout AUC ≈ 0.79).

### 2. Generate adversarial edge cases

```bash
python ml/edge_cases.py            # writes data/edge_cases.json
python ml/edge_cases.py --verify   # re-validates the generated file
```

### 3. Run the audit UI

```bash
streamlit run app.py
```

Pick a case in the sidebar, optionally enable **Use Mock LLM (Fast)**, and run
the audit. The trace, verdict banner, and audit log update live.

### 4. Run a single case from the CLI

```bash
python agent/graph.py
```

This calls the real Nightshift/Nemotron endpoint. For an offline run, use the
Streamlit **Mock LLM** toggle, or set `initial_state["use_mock"] = True` after
calling `state_for_edge_case(...)`.

### 5. Load the dataset into Zetaris

```bash
python scripts/load_to_zetaris.py
```

Creates the Lightning database/table, exports the CSV to Parquet, then verifies
the row count (1000) and the `amount` total.

### 6. Check connectivity

```bash
python scripts/smoke_test.py
```

## Environment variables

Copy `.env.example` to `.env` (never commit `.env`).

| Variable | Required | Purpose |
| --- | --- | --- |
| `ZETARIS_REST_URL` | yes | Zetaris REST base URL |
| `ZETARIS_ORG_ID` | yes | Zetaris organization ID |
| `ZETARIS_API_KEY` | yes | Zetaris API key |
| `MODEL_BASE_URL` | yes | Nightshift OpenAI-compatible base URL |
| `MODEL_API_KEY` | yes | Nightshift API key |
| `NEMOTRON_MODEL` | no | Model id (default `meterless/auto`) |
| `ZETARIS_DATABASE` | no | Target database (default `german_credit_db`) |
| `ZETARIS_TABLE_NAME` | no | Target table (default `german_credit`) |
| `ZETARIS_LOCAL_STORAGE_ROOT` | no | Zetaris storage root |
| `ZETARIS_EXPORT_PATH` | no | Parquet export path |
| `ZETARIS_BASE_URL` | no | Web UI URL used by the smoke test |

## Design notes

- **No hardcoded secrets.** All credentials come from `.env`, which is ignored
  by git.
- **Deterministic, not LLM-as-judge.** The evaluator is rule-based Python, so
  audit outcomes are reproducible.
- **Protected attributes** (`sex`, `age`) are derived explicitly in
  `ml/features.py` so bias flips can be detected.
- **Bounded autonomy.** Rollbacks are capped (`MAX_ROLLBACKS = 2`); after that
  the case is escalated rather than looping forever.
- **Zetaris audit log is mocked in the UI.** `agent/tools.py:log_to_zetaris`
  is a stub; the Streamlit log is in-memory. Wire it to the real table for
  production use.

## Limitations

- The bundled model and edge cases are for demonstration, not lending decisions.
- The Zetaris audit write path and Meterless memory retrieval are stubs.
- The system is not hardened for production banking security or real-time core
  banking integrations.

## License

No license has been added yet. Add one before distributing the project.
