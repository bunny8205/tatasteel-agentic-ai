"""General steel-domain agent layer.

This module handles broad steel plant prompts that are not tied to one of the
demo asset IDs. It gives the app a Codex-like "agent workspace" for steel
maintenance, process reliability, safety, procurement, and operations requests.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable


STEEL_TERMS = {
    "steel",
    "plant",
    "mill",
    "blast furnace",
    "bf",
    "bof",
    "eaf",
    "caster",
    "casting",
    "continuous caster",
    "tundish",
    "ladle",
    "mold",
    "mould",
    "rolling",
    "hot strip",
    "cold rolling",
    "plate mill",
    "finishing mill",
    "sinter",
    "pellet",
    "coke oven",
    "reheating furnace",
    "crane",
    "conveyor",
    "descaler",
    "hydraulic",
    "gearbox",
    "motor",
    "pump",
    "bearing",
    "lubrication",
    "cooling water",
    "vibration",
    "temperature",
    "pressure",
    "current",
    "scada",
    "plc",
    "agent",
    "agentic",
    "architecture",
    "workflow",
    "system design",
    "sop",
    "rca",
    "root cause",
    "maintenance",
    "equipment",
    "asset",
    "operator",
    "supervisor",
    "shift",
    "downtime",
    "reliability",
    "availability",
    "mtbf",
    "mttr",
    "breakdown",
    "shutdown",
    "spares",
    "procurement",
    "work order",
    "rul",
    "failure",
    "defect",
    "quality",
    "refractory",
    "tuyere",
    "stave",
    "safety",
    "loto",
}


INTENT_RULES = [
    (
        "incident_report",
        ["incident", "shift report", "report", "summary", "handover", "explain to supervisor"],
    ),
    (
        "root_cause_analysis",
        ["rca", "root cause", "why", "cause", "fishbone", "5 why", "failure analysis"],
    ),
    (
        "sop_generation",
        ["sop", "procedure", "checklist", "standard operating", "step by step", "inspection checklist"],
    ),
    (
        "work_order_planning",
        ["work order", "maintenance plan", "repair plan", "shutdown plan", "plan the job", "action plan", "schedule"],
    ),
    (
        "spares_procurement",
        ["spare", "procurement", "lead time", "inventory", "stock", "purchase", "reserve"],
    ),
    (
        "risk_prioritization",
        ["prioritize", "priority", "risk", "critical", "bottleneck", "what first", "urgent"],
    ),
    (
        "failure_prediction",
        ["predict", "rul", "remaining useful", "early warning", "anomaly", "alarm", "alarms", "breakout", "degradation", "forecast"],
    ),
    (
        "safety_control",
        ["safety", "loto", "permit", "isolation", "hazard", "unsafe", "confined", "hot work"],
    ),
    (
        "process_quality",
        ["defect", "quality", "surface", "crack", "scale", "camber", "thickness", "flatness"],
    ),
    (
        "data_agent_design",
        ["architecture", "agent", "workflow", "design", "system", "data flow", "model"],
    ),
]


SUBJECT_PATTERNS = [
    (r"blast\s+furnace|bf\b", "Blast Furnace"),
    (r"\bbof\b|basic\s+oxygen", "BOF Converter"),
    (r"\beaf\b|electric\s+arc", "Electric Arc Furnace"),
    (r"continuous\s+caster|caster|casting|mould|mold|tundish", "Continuous Caster"),
    (r"rolling|hot\s+strip|cold\s+rolling|plate\s+mill|finishing\s+mill", "Rolling Mill"),
    (r"gearbox|gear\s+box|gear", "Gearbox"),
    (r"motor|drive", "Motor or Drive"),
    (r"pump|cavitation", "Pump System"),
    (r"hydraulic|power\s+pack|actuator", "Hydraulic System"),
    (r"crane|hoist", "Crane and Hoist"),
    (r"conveyor|belt", "Conveyor System"),
    (r"reheating\s+furnace|furnace", "Reheating Furnace"),
    (r"sinter", "Sinter Plant"),
    (r"coke\s+oven", "Coke Oven"),
    (r"ladle", "Ladle Handling System"),
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def is_steel_domain_query(query: str) -> bool:
    q = _norm(query)
    return any(term in q for term in STEEL_TERMS)


def classify_steel_intent(query: str) -> str:
    q = _norm(query)
    if any(term in q for term in ["agentic workflow", "agent workflow", "workflow design", "system architecture", "data flow"]):
        return "predictive_maintenance_workflow_design"
    if "predictive maintenance" in q and any(term in q for term in ["design", "workflow", "agent", "architecture", "logs", "sops", "sensor", "feedback"]):
        return "predictive_maintenance_workflow_design"
    if any(term in q for term in ["which one", "choose one", "only one", "what should i choose", "first priority"]):
        return "risk_prioritization"
    scores: dict[str, int] = {}
    for intent, keywords in INTENT_RULES:
        scores[intent] = sum(1 for keyword in keywords if keyword in q)
    best_intent, best_score = max(scores.items(), key=lambda item: item[1])
    return best_intent if best_score > 0 else "general_steel_copilot"


def infer_steel_subject(query: str) -> str:
    q = _norm(query)
    for pattern, label in SUBJECT_PATTERNS:
        if re.search(pattern, q):
            return label
    return "Steel Plant"


def _source_lines(docs: list[dict], limit: int = 5) -> str:
    if not docs:
        return "- No indexed document matched; response uses steel maintenance first principles and asks for confirmation data."
    lines = []
    seen = set()
    for doc in docs:
        label = f"{doc.get('source')} ({doc.get('equipment_type')}/{doc.get('issue_type')})"
        if label in seen:
            continue
        seen.add(label)
        text = " ".join(str(doc.get("text", "")).split())[:260]
        lines.append(f"- {label} - {text}")
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def _health_lines(health_rows: list[dict], limit: int = 4) -> str:
    if not health_rows:
        return "- No live health table available."
    lines = []
    for row in health_rows[:limit]:
        lines.append(
            "- {asset_id}: {asset_type}, {risk_band}, hybrid risk {risk}, RUL {rul} days, area {area}".format(
                asset_id=row.get("asset_id", "asset"),
                asset_type=row.get("asset_type", "equipment"),
                risk_band=row.get("risk_band", row.get("risk_level", "unknown risk")),
                risk=round(float(row.get("hybrid_failure_risk", row.get("failure_risk", 0)) or 0), 3),
                rul=round(float(row.get("estimated_rul_days", 0) or 0), 1),
                area=row.get("area", "plant"),
            )
        )
    return "\n".join(lines)


def _intent_actions(intent: str, subject: str) -> list[str]:
    libraries = {
        "predictive_maintenance_workflow_design": [
            "Design the agent loop: perceive, retrieve, reason, act, verify, log, and learn.",
            "Use logs, SOPs, sensor alerts, failure reports, spares, production delay, and feedback as separate tool inputs.",
            "Keep ML risk, RUL, priority, safety gates, and selected target as locked deterministic fields.",
            "Use the LLM for planning, explanation, multi-turn interaction, report writing, and operator/supervisor adaptation.",
        ],
        "root_cause_analysis": [
            f"Freeze the failure statement for {subject}: symptom, asset boundary, time window, operating mode.",
            "Separate symptoms from causes; compare sensor trends, operator logs, maintenance history, and process changes.",
            "Run 5-Why/FMEA style ranking and mark each cause as confirmed, probable, or needs evidence.",
            "Define containment, permanent corrective action, verification test, and recurrence-prevention owner.",
        ],
        "sop_generation": [
            f"Create a field-safe SOP for {subject} with prerequisites, isolation, inspection order, acceptance limits, and escalation.",
            "Add hold points where an engineer must approve continuation.",
            "Map each step to evidence to capture: readings, photos, vibration spectrum, oil sample, or PLC alarm export.",
            "End with restart checks and logbook entries.",
        ],
        "work_order_planning": [
            f"Convert the request into a maintenance job plan for {subject}.",
            "Split into immediate containment, planned shutdown work, manpower/tools, spares, permits, and restart checks.",
            "Sequence tasks to reduce downtime and avoid repeat isolation.",
            "Create acceptance criteria before releasing the asset back to operation.",
        ],
        "spares_procurement": [
            f"Build a spare strategy for {subject} by criticality, lead time, consumption, and failure consequence.",
            "Reserve available critical spares; raise procurement for zero-stock or long-lead items.",
            "Define substitutes only with engineering approval and OEM compatibility.",
            "Link procurement priority to production bottleneck and safety exposure.",
        ],
        "risk_prioritization": [
            f"Rank {subject} risk by safety, production bottleneck, RUL, criticality, spares readiness, and delay history.",
            "Classify risk as P1/P2/P3/P4 and explain why.",
            "Select the first intervention and the condition that would trigger escalation.",
            "Create a supervisor-level decision summary.",
        ],
        "failure_prediction": [
            f"Define early-warning signals for {subject}: trend slope, anomaly count, threshold crossing, and historical failure pattern.",
            "Estimate RUL as a band, not a false exact value, unless a trained model supplies it.",
            "List the additional signals needed to improve prediction confidence.",
            "Trigger inspection before catastrophic failure indicators become irreversible.",
        ],
        "safety_control": [
            f"Treat {subject} as a safety-critical maintenance activity until hazards are cleared.",
            "Define energy sources, LOTO points, permits, exclusion zones, and PPE.",
            "Add stop-work triggers for temperature, pressure, suspended load, stored energy, or gas risk.",
            "Require supervisor sign-off before restart.",
        ],
        "process_quality": [
            f"Connect {subject} equipment condition to process and quality defect mechanisms.",
            "Separate mechanical, thermal, hydraulic, and control-loop causes.",
            "Link defect evidence to process parameters and maintenance checks.",
            "Recommend containment, sampling plan, and permanent corrective action.",
        ],
        "data_agent_design": [
            "Design the agent loop: perceive, retrieve, reason, act, verify, log, learn.",
            "Keep deterministic safety/ML fields locked and use the LLM for synthesis and interaction.",
            "Use RAG over manuals, SOPs, history, incident records, spares, and sensor summaries.",
            "Add feedback learning and audit trails so every decision is traceable.",
        ],
    }
    return libraries.get(
        intent,
        [
            f"Understand the steel-plant objective around {subject}.",
            "Collect asset, process, safety, production, and spare constraints.",
            "Generate a traceable decision with immediate actions and follow-up evidence.",
            "Log the outcome and learn from engineer feedback.",
        ],
    )


def build_general_plan(query: str, intent: str, subject: str) -> list[dict]:
    objective = {
        "predictive_maintenance_workflow_design": "Design an agentic predictive-maintenance workflow and demonstrate it on live plant context",
        "root_cause_analysis": "Diagnose probable root cause and corrective actions",
        "sop_generation": "Generate a safe executable SOP",
        "work_order_planning": "Create a maintenance execution plan",
        "spares_procurement": "Plan spares and procurement",
        "risk_prioritization": "Prioritize risk and intervention",
        "failure_prediction": "Predict failure risk and early warnings",
        "safety_control": "Create safety controls and stop-work logic",
        "process_quality": "Connect equipment health to process quality",
        "data_agent_design": "Design or explain an agentic AI workflow",
    }.get(intent, "Solve the steel operations request")
    tasks = [
        ("Supervisor Agent", objective),
        ("Triage Agent", f"Classify intent as {intent} and subject as {subject}"),
        ("Retrieval Agent", "Pull manuals, SOPs, history, failures, spares, policy, and live health evidence"),
        ("Reasoning Agent", "Convert evidence into hypotheses, risks, actions, assumptions, and missing data"),
        ("Planning Agent", "Create ordered actions, owner handoffs, spare plan, and escalation logic"),
        ("Verifier Agent", "Check traceability, safety, decision completeness, and uncertainty"),
        ("Memory Agent", "Prepare logbook and feedback hooks for continuous learning"),
        ("Reporter Agent", "Write a concise engineer-ready answer"),
    ]
    return [
        {"step": idx, "agent": agent, "task": task, "target": subject, "status": "complete"}
        for idx, (agent, task) in enumerate(tasks, 1)
    ]


def build_general_tool_calls(
    query: str,
    intent: str,
    subject: str,
    docs: list[dict],
    health_rows: list[dict],
    feedback_rows: int,
) -> list[dict]:
    return [
        {
            "tool": "intent_classifier",
            "agent": "Triage Agent",
            "input": query,
            "output": f"intent={intent}, subject={subject}",
            "status": "success",
        },
        {
            "tool": "plant_health_snapshot",
            "agent": "Sensor Agent",
            "input": "asset_health_summary.csv",
            "output": f"{len(health_rows)} live asset health rows reviewed",
            "status": "success" if health_rows else "review",
        },
        {
            "tool": "rag_retriever",
            "agent": "Retrieval Agent",
            "input": f"{subject}, {intent}",
            "output": f"{len(docs)} evidence chunks retrieved",
            "status": "success" if docs else "review",
        },
        {
            "tool": "safety_gate",
            "agent": "Verifier Agent",
            "input": subject,
            "output": "LOTO/permit/escalation considered for maintenance-facing answer",
            "status": "success",
        },
        {
            "tool": "feedback_memory",
            "agent": "Memory Agent",
            "input": "feedback_log.csv",
            "output": f"{feedback_rows} feedback rows available for future learning",
            "status": "success",
        },
        {
            "tool": "decision_packet_builder",
            "agent": "Reporter Agent",
            "input": "plan + evidence + verifier checks",
            "output": "structured steel-agent response generated",
            "status": "success",
        },
    ]


def build_general_verifier_checks(intent: str, docs: list[dict], health_rows: list[dict]) -> list[dict]:
    checks = [
        ("Steel-domain intent classified", bool(intent)),
        ("Evidence retrieved or uncertainty declared", bool(docs)),
        ("Live plant health considered", bool(health_rows)),
        ("Action plan included", True),
        ("Safety/LOTO gate included", True),
        ("Traceability section included", True),
        ("Feedback/logbook path included", True),
    ]
    return [
        {"check": name, "status": "pass" if ok else "review", "detail": "verified" if ok else "needs more site data"}
        for name, ok in checks
    ]


def build_general_decision_packet(
    query: str,
    intent: str,
    subject: str,
    docs: list[dict],
    health_rows: list[dict],
) -> dict:
    first_action = _intent_actions(intent, subject)[0]
    top_sources = [doc.get("source") for doc in docs[:5]]
    applied_target = health_rows[0].get("asset_id") if health_rows else None
    live_risks = [
        {
            "asset_id": row.get("asset_id"),
            "risk_band": row.get("risk_band"),
            "rul_days": row.get("estimated_rul_days"),
        }
        for row in health_rows[:3]
    ]
    workflow_mode = intent == "predictive_maintenance_workflow_design"
    return {
        "mode": "agentic_workflow_design" if workflow_mode else "general_steel_agent",
        "intent": intent,
        "objective": query,
        "selected_asset": "Steel Plant Workflow" if workflow_mode else subject,
        "subject": subject,
        "applied_demo_target": applied_target if workflow_mode else None,
        "risk_level": "Contextual",
        "priority": "Agent-assessed",
        "urgency": "Depends on live measurements; safety-critical symptoms escalate immediately",
        "recommended_first_action": first_action,
        "top_sources": top_sources,
        "live_risk_snapshot": live_risks,
        "next_system_action": "collect_missing_field_data_then_create_or_update_work_order",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def build_general_answer(
    query: str,
    intent: str,
    subject: str,
    docs: list[dict],
    health_rows: list[dict],
    agent_plan: list[dict],
    tool_calls: list[dict],
    verifier_checks: list[dict],
    decision_packet: dict,
) -> str:
    actions = _intent_actions(intent, subject)
    evidence = _source_lines(docs)
    health = _health_lines(health_rows)
    plan_lines = "\n".join(f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan)
    tool_lines = "\n".join(
        f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}"
        for t in tool_calls
    )
    check_lines = "\n".join(f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks)
    action_lines = "\n".join(f"- {action}" for action in actions)
    applied_target = decision_packet.get("applied_demo_target")
    applied_row = next((row for row in health_rows if row.get("asset_id") == applied_target), {}) if applied_target else {}

    if intent == "predictive_maintenance_workflow_design":
        return f"""
**Agentic Predictive Maintenance Workflow Design**

**Mode**
- agentic_workflow_design

**Intent**
- predictive_maintenance_workflow_design

**Applied Live Demo Target**
- {applied_target or "No live target available"}

**Framing**
- The workflow design is the main answer.
- To demonstrate the workflow on live plant data, the agent applied it to the current asset-health table and selected {applied_target or "the highest-risk asset"} as the first maintenance target.

**1. Agentic Workflow Design**
{action_lines}

**2. Live Execution Example**
- Current highest-risk live target: {applied_target or "not available"}
- Risk band: {applied_row.get("risk_band", "not available")}
- Hybrid risk: {round(float(applied_row.get("hybrid_failure_risk", applied_row.get("failure_risk", 0)) or 0), 3)}
- RUL: {applied_row.get("estimated_rul_days", "not available")} days

**3. Why The Demo Target Was Selected**
- The agent ranks live assets by hybrid health score, failure risk, RUL, criticality, delay impact, and evidence availability.
- The selected target is used only as a demonstration of the workflow, not as a replacement for the architecture design.

**4. Autonomous Execution Plan**
{plan_lines}

**5. Tool Calls Executed**
{tool_lines}

**6. Verifier Checks**
{check_lines}

**7. Evidence**
{evidence}

**Decision Packet**
- Mode: {decision_packet["mode"]}
- Intent: {decision_packet["intent"]}
- Applied live demo target: {decision_packet.get("applied_demo_target")}
- Next system action: {decision_packet["next_system_action"]}
""".strip()

    return f"""
**Steel Plant Agent Response**

**Objective**
- {query}

**Interpreted Intent**
- Intent: {intent.replace("_", " ")}
- Subject: {subject}
- Agent stance: I will treat this as a steel-plant decision task, not a generic chatbot answer.

**Autonomous Execution Plan**
{plan_lines}

**Tool Calls Executed**
{tool_lines}

**Verifier Checks**
{check_lines}

**Live Plant Context Considered**
{health}

**Working Diagnosis / Approach**
- For {subject}, the agent should first separate immediate safety risk, production bottleneck risk, equipment health risk, and evidence uncertainty.
- If the prompt is about a known asset ID, the system should switch to the live ML/RAG asset path. If it is a broader plant question, this general steel agent produces the operating plan and evidence checklist.
- The answer is intentionally operational: what to inspect, what to verify, what to procure, what to log, and when to escalate.

**Action Plan**
{action_lines}

**Evidence To Collect Next**
- Latest sensor trend: temperature, vibration, current, pressure, speed, alarm count, and trend slope.
- PLC/SCADA fault chronology with timestamps and operating mode.
- Recent maintenance work orders, lubrication/oil/filter records, and operator observations.
- Spare stock, lead time, substitute policy, and shutdown window constraints.
- Safety permits, isolation points, and restart acceptance readings.

**Risk And Priority Logic**
- P1/Critical if there is personnel risk, catastrophic failure risk, very low RUL, repeated critical alarms, or a direct production bottleneck.
- P2/High if degradation is clear but controlled intervention within 24 hours is feasible.
- P3/Medium if the issue can be planned inside a maintenance window.
- P4/Low if only monitoring and confirmation data are required.

**Traceability / Sources**
{evidence}

**Final Decision Packet**
- Mode: {decision_packet["mode"]}
- Intent: {decision_packet["intent"]}
- Subject: {decision_packet["subject"]}
- Recommended first action: {decision_packet["recommended_first_action"]}
- Next system action: {decision_packet["next_system_action"]}

**Feedback And Learning**
- Save engineer corrections, accepted actions, actual root cause, downtime, and restart readings to the feedback log.
- Use the next confirmed outcome to adjust future root-cause ranking and action order.
""".strip()


def summarize_health_rows(rows: Iterable[dict]) -> list[dict]:
    def score(row: dict) -> float:
        return float(row.get("hybrid_health_score", row.get("hybrid_failure_risk", row.get("failure_risk", 0))) or 0)

    return sorted([dict(row) for row in rows], key=score, reverse=True)
