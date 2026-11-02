# OpenCode Budget Rules

## Plan gate
Before editing any file:
1. Write a plan listing files to change, intended changes, tests, and rollback/isolation.
2. Wait for explicit approval: "APPROVED".
3. Do not write code before approval.
4. If the plan changes, stop and request re-approval.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.

## RTK

RTK (Rust Token Killer) is installed and its OpenCode plugin rewrites whole recognized shell commands (e.g. a bare `git status`) to a compressed form automatically. It saves tokens on command output; graphify saves tokens on code reading. Use both.

Rules:
- Prefer the explicit RTK forms when you expect large output: `rtk read <file>`, `rtk grep <pattern>`, `rtk git diff`, `rtk test <cmd>`, `rtk log <file>`, `rtk json <file>`.
- The hook only matches whole commands. Pipes, unrecognized flags, and PowerShell-only cmdlets bypass it — use the explicit `rtk` form when the output would otherwise be large.
- Never set `RTK_DISABLED=1`, and do not add it to `.env` or any shell profile.
- Do not pipe RTK output through truncation helpers; it is already compact.
- If `rtk gain` warns `No hook installed`, the plugin is not loading — check that `rtk` is still on PATH before assuming savings.
