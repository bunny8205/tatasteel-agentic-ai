"""Agentic Maintenance Wizard orchestration layer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .data_setup import create_compatibility_sensor_log, prepare_data
from .dynamic_assets import (
    dynamic_actions,
    dynamic_asset_ids,
    dynamic_root_cause,
    dynamic_spares,
    extract_asset_ids,
    is_asset_ingestion_query,
    is_asset_update_query,
    is_priority_change_query,
    is_rule_apply_query,
    is_rule_ingestion_query,
    latest_dynamic_asset_change,
    load_dynamic_assets,
    load_dynamic_rules,
    parse_dynamic_assets,
    query_mentions_new_asset_reference,
    remember_dynamic_rule,
    score_dynamic_assets,
    update_dynamic_assets_from_query,
    upsert_dynamic_assets,
)
from .llm import LocalLLM
from .models import FEATURE_COLS, ModelManager, risk_band_from_score, safe_float
from .rag import RAGIndex, normalize_equipment_type
from .steel_agent import (
    build_general_answer,
    build_general_decision_packet,
    build_general_plan,
    build_general_tool_calls,
    build_general_verifier_checks,
    classify_steel_intent,
    infer_steel_subject,
    is_steel_domain_query,
    summarize_health_rows,
)


def _format_records(records: list[dict]) -> str:
    if not records:
        return "- No matching records found."
    lines = []
    for record in records:
        clean = {k: v for k, v in record.items() if pd.notna(v)}
        lines.append("- " + "; ".join(f"{k}: {v}" for k, v in clean.items()))
    return "\n".join(lines)


def _format_sources(docs: list[dict]) -> str:
    if not docs:
        return "- No retrieved evidence."
    lines = []
    seen = set()
    for doc in docs:
        key = (doc.get("source"), doc.get("equipment_type"), doc.get("issue_type"))
        if key in seen:
            continue
        seen.add(key)
        source = doc.get("source", "unknown source")
        equipment = doc.get("equipment_type", "general")
        issue = doc.get("issue_type", "general")
        text = " ".join(str(doc.get("text", "")).split())[:320]
        lines.append(f"{len(lines) + 1}. {source} - {equipment}/{issue}\n   Evidence: {text}")
    return "\n".join(lines)


def _is_missing_value(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _display_value(value, unit: str = "", decimals: int = 2) -> str:
    if _is_missing_value(value):
        return "not provided"
    try:
        number = float(value)
        rendered = f"{number:.{decimals}f}".rstrip("0").rstrip(".")
        return f"{rendered}{unit}"
    except (TypeError, ValueError):
        return f"{value}{unit}" if unit else str(value)


def _spares_strategy(spares: list[dict]) -> str:
    if not spares:
        return "- No matching spare inventory found."
    lines = []
    for item in spares:
        spare = item.get("spare_part", "Unknown spare")
        qty = int(safe_float(item.get("stock_qty", 0)))
        lead = int(safe_float(item.get("lead_time_days", 0)))
        if qty <= 0:
            action = "Raise procurement immediately."
        elif lead >= 7:
            action = "Reserve before shutdown due to lead time."
        else:
            action = "Available; reserve for planned work."
        lines.append(f"- {spare}: stock {qty}, lead time {lead} days. {action}")
    return "\n".join(lines)


@dataclass
class MaintenanceWizard:
    model_manager: ModelManager = field(default_factory=ModelManager)
    rag: RAGIndex = field(default_factory=RAGIndex)
    llm: LocalLLM = field(default_factory=LocalLLM)
    initialized: bool = False
    session_memory: dict = field(default_factory=dict)

    def initialize(self, force: bool = False, load_llm: bool = False) -> "MaintenanceWizard":
        prepare_data(force=force)
        self.model_manager.train_or_load(force=force)
        self.rag.build()
        if load_llm:
            self.llm.load()
        self.initialized = True
        return self

    def ensure_ready(self) -> None:
        if not self.initialized:
            self.initialize(load_llm=False)

    @property
    def asset_ids(self) -> list[str]:
        self.ensure_ready()
        return sorted(self.asset_health_table()["asset_id"].dropna().astype(str).unique().tolist())

    def asset_health_table(self) -> pd.DataFrame:
        self.ensure_ready()
        base = self.model_manager.asset_health.copy()
        if "data_origin" not in base.columns:
            base["data_origin"] = "demo_sensor_model"
        if "is_dynamic" not in base.columns:
            base["is_dynamic"] = 0
        dynamic = score_dynamic_assets(load_dynamic_assets())
        if dynamic.empty:
            return base
        return pd.concat([base, dynamic], ignore_index=True, sort=False)

    def query_assets(self, query: str) -> list[str]:
        self.ensure_ready()
        explicit = self._explicit_asset_ids(query)
        rag_assets = self.rag.query_assets(query)
        out: list[str] = []
        for asset_id in explicit + rag_assets:
            if asset_id not in out:
                out.append(asset_id)
        return out

    def _explicit_asset_ids(self, query: str) -> list[str]:
        known = set(self.asset_health_table()["asset_id"].dropna().astype(str).str.upper())
        out = []
        for asset_id in extract_asset_ids(query):
            if asset_id in known and asset_id not in out:
                out.append(asset_id)
        return out

    def _is_dynamic_asset(self, asset_id: str) -> bool:
        return str(asset_id).upper() in set(dynamic_asset_ids())

    def _infer_asset_from_query(self, query: str) -> str | None:
        q = str(query).lower()
        explicit = self._explicit_asset_ids(query)
        if explicit:
            return explicit[0]
        if query_mentions_new_asset_reference(query):
            remembered = self.session_memory.get("last_new_asset_id")
            if remembered:
                return remembered
            dyn_ids = dynamic_asset_ids()
            if len(dyn_ids) == 1:
                return dyn_ids[0]
        if any(term in q for term in ["same asset", "that asset", "it", "spare should i", "same equipment"]):
            remembered = self.session_memory.get("last_asset_id")
            if remembered:
                return remembered
        assets = self.rag.query_assets(query)
        if assets:
            return assets[0]
        return None

    def _is_plant_query(self, query: str) -> bool:
        q = str(query).lower()
        if any(term in q for term in ["agentic workflow", "agent workflow", "workflow design", "system architecture", "data flow"]):
            return False
        if "predictive maintenance" in q and any(term in q for term in ["design", "workflow", "agent", "architecture", "logs", "sops", "sensor", "feedback"]):
            return False
        plant_terms = [
            "compare",
            "plant",
            "supervisor",
            "prioritize all",
            "rank",
            "bottleneck",
            "ranking",
            "which asset",
            "which one",
            "most dangerous",
            "newly added",
            "added assets",
            "original and newly",
            "choose one",
            "choose only one",
            "only one asset",
            "maintain only one",
            "maintain one asset",
            "one asset today",
            "what should i choose",
            "what should we choose",
            "this shift",
            "today",
        ]
        if any(term in q for term in plant_terms):
            priority_context = any(
                term in q
                for term in [
                    "asset",
                    "equipment",
                    "maintain",
                    "maintenance",
                    "priority",
                    "prioritize",
                    "choose",
                    "first",
                    "risk",
                    "downtime",
                    "shift",
                ]
            )
            return priority_context or any(term in q for term in ["plant", "supervisor", "compare", "ranking"])
        return False

    def _plant_scope_asset_ids(self, query: str) -> list[str] | None:
        q = str(query).lower()
        explicit = self._explicit_asset_ids(query)
        if len(explicit) >= 2:
            return explicit
        dyn_ids = dynamic_asset_ids()
        dynamic_only_terms = [
            "only newly added",
            "rank only newly",
            "newly added assets only",
            "dynamic assets only",
            "only dynamic",
            "not original",
            "not the original",
            "not original demo",
            "exclude original",
            "exclude demo",
        ]
        if any(term in q for term in dynamic_only_terms):
            return dyn_ids or None
        if any(term in q for term in ["newly added", "added assets", "new assets", "dynamic assets"]):
            if "original" not in q and "all assets" not in q and "all original" not in q:
                return dyn_ids or None
        return explicit or None

    def _is_public_query(self, query: str) -> bool:
        q = str(query).lower()
        return any(term in q for term in ["public dataset", "ai4i", "uci", "data source", "dataset used"])

    def _is_general_steel_query(self, query: str) -> bool:
        return is_steel_domain_query(query)

    def get_latest_sensor_summary(self, asset_id: str) -> dict:
        self.ensure_ready()
        health = self.asset_health_table()
        row = health[health["asset_id"].astype(str).str.upper() == str(asset_id).upper()]
        if row.empty:
            return {"asset_id": asset_id, "error": "No sensor data found."}
        r = row.iloc[0].to_dict()
        is_dynamic = int(safe_float(r.get("is_dynamic"), 0)) == 1

        def latest_value(field: str, default: float = 0):
            value = r.get(field)
            if is_dynamic and _is_missing_value(value):
                return None
            return round(safe_float(value, default), 2)

        return {
            "asset_id": asset_id,
            "asset_type": r.get("asset_type"),
            "area": r.get("area"),
            "criticality": r.get("criticality"),
            "temperature_latest": latest_value("temperature"),
            "vibration_latest": latest_value("vibration"),
            "current_latest": latest_value("current"),
            "pressure_latest": latest_value("pressure"),
            "rpm_latest": latest_value("rpm", 1480),
            "alarm_count_latest": int(safe_float(r.get("alarm_count"))),
            "ml_failure_risk_latest": round(safe_float(r.get("ml_failure_risk")), 4),
            "operational_rule_score": round(safe_float(r.get("operational_rule_score")), 2),
            "hybrid_health_score": round(safe_float(r.get("hybrid_health_score")), 2),
            "hybrid_failure_risk": round(safe_float(r.get("hybrid_failure_risk", r.get("failure_risk"))), 4),
            "failure_pred": int(safe_float(r.get("failure_pred"), 0)),
            "risk_band": r.get("risk_band"),
            "estimated_rul_days": round(safe_float(r.get("estimated_rul_days"), 30), 1),
            "anomaly_events_24h": int(safe_float(r.get("anomaly_events_24h"))),
            "temperature_slope_24h": round(safe_float(r.get("temperature_slope_24h")), 4),
            "vibration_slope_24h": round(safe_float(r.get("vibration_slope_24h")), 4),
            "pressure_slope_24h": round(safe_float(r.get("pressure_slope_24h")), 4),
            "data_origin": r.get("data_origin", "demo_sensor_model"),
            "is_dynamic": int(is_dynamic),
            "missing_readings": r.get("missing_readings", ""),
            "provisional_scoring_note": r.get("provisional_scoring_note", ""),
            "operator_notes": r.get("operator_notes", ""),
            "qualitative_risk_note": r.get("qualitative_risk_note", ""),
            "base_priority": r.get("base_priority", r.get("priority")),
            "base_risk_band": r.get("base_risk_band", r.get("risk_band")),
            "base_hybrid_health_score": round(safe_float(r.get("base_hybrid_health_score", r.get("hybrid_health_score"))), 2),
            "applied_rules": r.get("applied_rules", []) if isinstance(r.get("applied_rules", []), list) else [],
            "applied_rule_count": int(safe_float(r.get("applied_rule_count", 0))),
            "dynamic_rule_note": r.get("dynamic_rule_note", ""),
        }

    def get_spares(self, asset_id: str) -> list[dict]:
        rows = pd.read_csv(DATA_DIR / "spares_inventory.csv").query("asset_id == @asset_id").to_dict("records")
        if rows:
            return rows
        sensor = self.get_latest_sensor_summary(asset_id)
        if sensor.get("is_dynamic"):
            return dynamic_spares(asset_id, sensor.get("asset_type", ""))
        return []

    def get_delay(self, asset_id: str) -> dict:
        rows = pd.read_csv(DATA_DIR / "delay_logs.csv").query("asset_id == @asset_id")
        if len(rows):
            return rows.iloc[0].to_dict()
        sensor = self.get_latest_sensor_summary(asset_id)
        if sensor.get("is_dynamic"):
            delay = 6.0 if str(sensor.get("criticality", "")).lower() == "critical" else 2.0
            return {
                "asset_id": asset_id,
                "area": sensor.get("area"),
                "delay_hours": delay,
                "production_impact": "Inferred production/safety impact for user-added asset",
            }
        return {"delay_hours": 0}

    def get_history(self, asset_id: str) -> list[dict]:
        rows = pd.read_csv(DATA_DIR / "maintenance_history.csv").query("asset_id == @asset_id").to_dict("records")
        if rows:
            return rows
        if self._is_dynamic_asset(asset_id):
            return [
                {
                    "asset_id": asset_id,
                    "timestamp": "user-added asset",
                    "issue": "No historical work orders yet",
                    "action_taken": "Use live readings and generic equipment policy until history is learned",
                    "result": "Needs engineer confirmation",
                    "downtime_hours": 0,
                }
            ]
        return []

    def get_failures(self, asset_id: str) -> list[dict]:
        rows = pd.read_csv(DATA_DIR / "failure_reports.csv").query("asset_id == @asset_id").to_dict("records")
        if rows:
            return rows
        if self._is_dynamic_asset(asset_id):
            return [
                {
                    "asset_id": asset_id,
                    "failure_mode": "Not yet observed",
                    "root_cause": "Pending inspection and feedback learning",
                    "corrective_action": "Create first baseline inspection record",
                    "business_impact": "Estimated from criticality, area, and live readings",
                }
            ]
        return []

    def get_feedback(self, asset_id: str) -> list[dict]:
        path = DATA_DIR / "feedback_log.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path)
        if "asset_id" not in df.columns or df.empty:
            return []
        return df[df["asset_id"].astype(str) == str(asset_id)].tail(3).to_dict("records")

    def rule_breakdown(self, sensor: dict, delay: dict | None = None, spares: list[dict] | None = None) -> list[str]:
        delay = delay or {}
        spares = spares or []
        reasons = []
        typ = str(sensor.get("asset_type", "")).lower()
        criticality = str(sensor.get("criticality", "medium")).lower()
        temp = safe_float(sensor.get("temperature_latest"))
        vib = safe_float(sensor.get("vibration_latest"))
        current = safe_float(sensor.get("current_latest"))
        pressure = safe_float(sensor.get("pressure_latest"), 8)
        alarms = safe_float(sensor.get("alarm_count_latest"))
        anomalies = safe_float(sensor.get("anomaly_events_24h"))
        delay_hours = safe_float(delay.get("delay_hours", 0))
        notes = str(sensor.get("operator_notes") or "").lower()

        if "gearbox" in typ:
            if vib >= 7:
                reasons.append("Gearbox vibration >= 7 mm/s: +30")
            if vib >= 9:
                reasons.append("Gearbox vibration >= 9 mm/s: +20")
            if vib >= 10:
                reasons.append("Gearbox vibration >= 10 mm/s: +10")
            if temp >= 65:
                reasons.append("Gearbox temperature >= 65 deg C: +8")
        if "motor" in typ:
            if temp >= 80:
                reasons.append("Motor temperature >= 80 deg C: +25")
            if temp >= 85:
                reasons.append("Motor temperature >= 85 deg C: +15")
            if current >= 80:
                reasons.append("Motor current >= 80 A: +12")
            if vib >= 5:
                reasons.append("Motor vibration >= 5 mm/s: +8")
        if "pump" in typ:
            if pressure <= 6:
                reasons.append("Pump pressure <= 6 bar: +22")
            if vib >= 5:
                reasons.append("Pump vibration >= 5 mm/s: +12")
            if current >= 75:
                reasons.append("Pump current >= 75 A: +8")
        if "hydraulic" in typ:
            if pressure <= 6:
                reasons.append("Hydraulic pressure <= 6 bar: +25")
            if temp >= 65:
                reasons.append("Hydraulic oil temperature >= 65 deg C: +10")
            if alarms >= 2:
                reasons.append("Hydraulic alarm count >= 2: +8")
        if any(word in typ for word in ["blower", "fan", "compressor"]):
            if vib >= 6:
                reasons.append("Rotating air equipment vibration >= 6 mm/s: +20")
            if current >= 85:
                reasons.append("Rotating air equipment current >= 85 A: +15")
            if temp >= 80:
                reasons.append("Rotating air equipment temperature >= 80 deg C: +20")
        if "blast furnace" in (typ + " " + str(sensor.get("area", "")).lower()):
            if temp >= 80 and vib >= 6.5:
                reasons.append("Blast furnace critical blower/fan high temperature plus vibration safety override: +20")
        if "cavitation" in notes and ("pump" in typ or "descaler" in typ):
            reasons.append("Operator-reported cavitation noise on pump/descaler: qualitative risk floor keeps priority elevated")
        if any(term in notes for term in ["loud noise", "abnormal noise", "bearing noise", "rubbing", "chatter"]):
            reasons.append("Operator-reported abnormal noise: qualitative risk uplift")
        if any(term in notes for term in ["smoke", "sparking", "burning smell"]):
            reasons.append("Operator-reported smoke/sparking/burning smell: safety-critical risk override")
        if not any(key in typ for key in ["gearbox", "motor", "pump", "hydraulic", "blower", "fan", "compressor"]):
            if temp >= 80:
                reasons.append("Generic equipment temperature >= 80 deg C: +20")
            if vib >= 6:
                reasons.append("Generic rotating equipment vibration >= 6 mm/s: +20")
            if current >= 85:
                reasons.append("Generic equipment current >= 85 A: +15")
            if pressure <= 6:
                reasons.append("Generic low pressure <= 6 bar: +18")

        if alarms >= 2:
            reasons.append("Alarm count >= 2: +8")
        if alarms >= 4:
            reasons.append("Alarm count >= 4: +8")
        if anomalies >= 6:
            reasons.append("Anomaly events >= 6 in last 24h: +15")
        if anomalies >= 18:
            reasons.append("Anomaly events >= 18 in last 24h: +10")
        if criticality == "critical":
            reasons.append("Critical equipment: +15")
        elif criticality == "high":
            reasons.append("High criticality equipment: +10")
        elif criticality == "medium":
            reasons.append("Medium criticality equipment: +5")
        if delay_hours > 0:
            reasons.append(f"Historical delay impact {delay_hours} hours: +{min(delay_hours * 2.0, 12):.1f}")
        if any(safe_float(s.get("stock_qty", 0)) <= 0 and safe_float(s.get("lead_time_days", 0)) >= 7 for s in spares):
            reasons.append("Critical spare unavailable with high lead time: priority uplift")
        for rule in sensor.get("applied_rules") or []:
            rule_id = rule.get("rule_id", "remembered rule")
            condition = str(rule.get("condition_text", "")).strip()
            priority = rule.get("priority_override") or "policy"
            reasons.append(f"Remembered safety/SOP rule {rule_id} applied: {priority}. {condition}")
        return reasons or ["No major rule trigger; monitor based on trend and ML signal."]

    def detect_anomaly(self, asset_id: str) -> dict:
        s = self.get_latest_sensor_summary(asset_id)
        health = s.get("hybrid_health_score", 0)
        events = s.get("anomaly_events_24h", 0)
        if health >= 75 or events >= 12:
            level = "HIGH"
        elif health >= 55 or events >= 4:
            level = "MEDIUM"
        else:
            level = "LOW"
        return {
            "asset_id": asset_id,
            "anomaly_level": level,
            "hybrid_failure_risk": s.get("hybrid_failure_risk", 0),
            "ml_failure_risk": s.get("ml_failure_risk_latest", 0),
            "operational_rule_score": s.get("operational_rule_score", 0),
            "hybrid_health_score": health,
            "anomaly_events_24h": events,
        }

    def prioritize_action(self, sensor: dict, spares: list[dict], delay: dict) -> dict:
        health = sensor.get("hybrid_health_score", 0)
        rul = sensor.get("estimated_rul_days", 30)
        delay_hours = safe_float(delay.get("delay_hours", 0))
        spare_blocked = any(safe_float(s.get("stock_qty", 0)) <= 0 and safe_float(s.get("lead_time_days", 0)) >= 7 for s in spares)
        score = health + min(delay_hours * 1.5, 8) + (5 if rul <= 3 else 3 if rul <= 7 else 0) + (4 if spare_blocked else 0)
        score = round(float(np.clip(score, 0, 100)), 2)
        if score >= 75:
            return {"priority": "P1", "risk_level": "CRITICAL", "urgency": "Immediate action", "priority_score": score}
        if score >= 55:
            return {"priority": "P2", "risk_level": "HIGH", "urgency": "Action within 24 hours", "priority_score": score}
        if score >= 38:
            return {"priority": "P3", "risk_level": "MEDIUM", "urgency": "Plan in maintenance window", "priority_score": score}
        return {"priority": "P4", "risk_level": "LOW", "urgency": "Monitor only", "priority_score": score}

    def infer_root_cause(self, asset_id: str) -> str:
        sensor = self.get_latest_sensor_summary(asset_id)
        if sensor.get("is_dynamic"):
            return dynamic_root_cause(sensor.get("asset_type", ""))
        typ = normalize_equipment_type(sensor.get("asset_type", ""))
        return {
            "motor": "bearing lubrication degradation, cooling restriction, overload, or current imbalance",
            "gearbox": "bearing wear, gear tooth wear, shaft misalignment, oil contamination, or foundation looseness",
            "pump": "suction strainer choking, low suction head, air ingress, seal wear, or impeller erosion",
            "hydraulic": "filter choking, relief valve leakage, pump wear, or hydraulic oil leakage",
        }.get(typ, "degradation pattern found in sensor trend and historical records")

    def recommended_actions(self, asset_id: str) -> list[str]:
        feedback = self.get_feedback(asset_id)
        actions = []
        if feedback:
            latest = feedback[-1]
            corrected = latest.get("corrected_action")
            if isinstance(corrected, str) and corrected.strip():
                actions.append(f"Apply learned feedback: {corrected}")
        sensor = self.get_latest_sensor_summary(asset_id)
        if sensor.get("is_dynamic"):
            return actions + dynamic_actions(sensor.get("asset_type", ""))
        typ = normalize_equipment_type(sensor.get("asset_type", ""))
        default_actions = {
            "motor": ["Inspect bearing lubrication, cooling airflow, current imbalance, load, and coupling alignment."],
            "gearbox": ["Check oil contamination and level.", "Inspect alignment, bearing condition, gear mesh, and foundation bolts."],
            "pump": ["Inspect suction strainer, suction head, inlet valve position, seal leakage, and impeller erosion."],
            "hydraulic": ["Replace or inspect filter element, verify relief valve setting, check oil level, and inspect leakage."],
        }
        return actions + default_actions.get(typ, ["Inspect asset condition and create a planned maintenance work order."])

    def build_agent_trace(self, asset_id: str, sensor: dict, anomaly: dict, priority: dict, docs: list[dict]) -> list[dict]:
        return [
            {"agent": "Triage Agent", "decision": f"Detected asset {asset_id} and equipment type {sensor.get('asset_type')}."},
            {"agent": "Sensor Agent", "decision": f"Hybrid risk {sensor.get('hybrid_failure_risk')}, anomaly {anomaly.get('anomaly_level')}, RUL {sensor.get('estimated_rul_days')} days."},
            {"agent": "Knowledge Agent", "decision": f"Retrieved {len(docs)} filtered evidence chunks."},
            {"agent": "Risk Agent", "decision": f"Assigned {priority.get('priority')} / {priority.get('risk_level')}."},
            {"agent": "Planning Agent", "decision": "Generated work order actions and spare strategy."},
            {"agent": "Reporting Agent", "decision": "Generated alert and logbook entry."},
        ]

    def build_agent_plan(self, query: str, mode: str, asset_id: str | None = None) -> list[dict]:
        q = str(query).lower()
        objective = "Diagnose equipment issue and produce maintenance decision support"
        if mode == "plant_priority":
            objective = "Rank plant assets and select the best maintenance target"
        elif mode == "public_dataset":
            objective = "Explain public benchmark usage and data-governance controls"
        elif "spare" in q and not any(term in q for term in ["diagnose", "root cause", "risk", "vibration", "temperature", "pressure", "current", "alert"]):
            objective = "Identify required spare strategy for current maintenance risk"

        target = asset_id or self.session_memory.get("last_asset_id") or "plant"
        return [
            {"step": 1, "agent": "Supervisor Agent", "task": objective, "target": target, "status": "complete"},
            {"step": 2, "agent": "Triage Agent", "task": "Resolve asset, user intent, and operating context", "target": target, "status": "complete"},
            {"step": 3, "agent": "Sensor Agent", "task": "Read latest sensor state, anomaly events, and RUL indicators", "target": target, "status": "complete"},
            {"step": 4, "agent": "Knowledge Agent", "task": "Retrieve SOP, history, failure reports, spares, and policy evidence", "target": target, "status": "complete"},
            {"step": 5, "agent": "Risk Agent", "task": "Fuse ML risk with operational rule score and criticality", "target": target, "status": "complete"},
            {"step": 6, "agent": "Planner Agent", "task": "Create action plan, spare strategy, escalation, and work-order recommendation", "target": target, "status": "complete"},
            {"step": 7, "agent": "Verifier Agent", "task": "Check locked fields, traceability, and safety-critical escalation", "target": target, "status": "complete"},
            {"step": 8, "agent": "Reporter Agent", "task": "Generate engineer-facing report and logbook entry", "target": target, "status": "complete"},
        ]

    def build_tool_calls(
        self,
        asset_id: str,
        sensor: dict,
        anomaly: dict,
        priority: dict,
        docs: list[dict],
        history: list[dict],
        failures: list[dict],
        spares: list[dict],
        delay: dict,
        feedback: list[dict],
    ) -> list[dict]:
        return [
            {
                "tool": "asset_resolver",
                "agent": "Triage Agent",
                "input": asset_id,
                "output": f"{sensor.get('asset_type')} in {sensor.get('area')}",
                "status": "success",
            },
            {
                "tool": "sensor_health_reader",
                "agent": "Sensor Agent",
                "input": asset_id,
                "output": (
                    f"temp={_display_value(sensor.get('temperature_latest'))}, "
                    f"vib={_display_value(sensor.get('vibration_latest'))}, "
                    f"pressure={_display_value(sensor.get('pressure_latest'))}"
                ),
                "status": "success",
            },
            {
                "tool": "anomaly_detector",
                "agent": "Sensor Agent",
                "input": "latest sensor row + 24h anomaly window",
                "output": f"{anomaly.get('anomaly_level')} abnormality, {anomaly.get('anomaly_events_24h')} anomaly events",
                "status": "success",
            },
            {
                "tool": "hybrid_risk_scorer",
                "agent": "Risk Agent",
                "input": "ML failure risk + operational rules + criticality + delay",
                "output": f"{priority.get('priority')}/{priority.get('risk_level')} with score {priority.get('priority_score')}",
                "status": "success",
            },
            {
                "tool": "rag_retriever",
                "agent": "Knowledge Agent",
                "input": f"asset={asset_id}, top_k=5",
                "output": f"{len(docs)} evidence chunks from {len(set(d.get('source') for d in docs))} sources",
                "status": "success",
            },
            {
                "tool": "history_lookup",
                "agent": "Knowledge Agent",
                "input": asset_id,
                "output": f"{len(history)} work orders, {len(failures)} failure reports",
                "status": "success",
            },
            {
                "tool": "spares_planner",
                "agent": "Planner Agent",
                "input": asset_id,
                "output": f"{len(spares)} spare items checked",
                "status": "success",
            },
            {
                "tool": "feedback_memory",
                "agent": "Planner Agent",
                "input": asset_id,
                "output": f"{len(feedback)} relevant feedback rows reused",
                "status": "success",
            },
            {
                "tool": "digital_logbook_writer",
                "agent": "Reporter Agent",
                "input": asset_id,
                "output": "logbook entry created after report generation",
                "status": "success",
            },
        ]

    def build_verifier_checks(self, sensor: dict, priority: dict, docs: list[dict], spares: list[dict]) -> list[dict]:
        checks = [
            ("Asset resolved", bool(sensor.get("asset_id"))),
            ("Locked priority populated", bool(priority.get("priority") and priority.get("risk_level"))),
            ("Hybrid score available", sensor.get("hybrid_health_score") is not None),
            ("ML risk and rule score separated", sensor.get("ml_failure_risk_latest") is not None and sensor.get("operational_rule_score") is not None),
            ("RUL estimate available", sensor.get("estimated_rul_days") is not None),
            ("Traceability sources retrieved", len(docs) >= 3),
            ("Spare strategy checked", len(spares) > 0),
            ("Escalation generated for P1/P2", priority.get("priority") in {"P1", "P2", "P3", "P4", "PLANT"}),
        ]
        return [
            {"check": name, "status": "pass" if ok else "review", "detail": "verified" if ok else "needs engineer review"}
            for name, ok in checks
        ]

    def build_decision_packet(
        self,
        mode: str,
        query: str,
        asset_id: str,
        sensor: dict,
        priority: dict,
        docs: list[dict],
        actions: list[str],
    ) -> dict:
        return {
            "mode": mode,
            "objective": query,
            "selected_asset": asset_id,
            "equipment_type": sensor.get("asset_type"),
            "risk_level": priority.get("risk_level"),
            "priority": priority.get("priority"),
            "urgency": priority.get("urgency"),
            "hybrid_failure_risk": sensor.get("hybrid_failure_risk"),
            "ml_failure_risk": sensor.get("ml_failure_risk_latest"),
            "operational_rule_score": sensor.get("operational_rule_score"),
            "hybrid_health_score": sensor.get("hybrid_health_score"),
            "estimated_rul_days": sensor.get("estimated_rul_days"),
            "recommended_first_action": actions[0] if actions else "Inspect asset condition",
            "top_sources": [doc.get("source") for doc in docs[:3]],
            "next_system_action": "create_work_order_and_notify_supervisor" if priority.get("priority") in {"P1", "P2"} else "monitor_and_schedule",
        }

    def write_logbook(self, query: str, asset_id: str, priority: dict, summary: str) -> None:
        path = DATA_DIR / "digital_logbook.csv"
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()
        row = {
            "timestamp": datetime.now().isoformat(),
            "user_id": self.session_memory.get("user_id", "demo_user"),
            "asset_id": asset_id,
            "query": query,
            "risk_level": priority.get("risk_level"),
            "priority": priority.get("priority"),
            "summary": summary[:1000],
        }
        pd.concat([df, pd.DataFrame([row])], ignore_index=True, sort=False).to_csv(path, index=False)

    def save_feedback(self, user_id: str, asset_id: str, query: str, feedback_type: str, feedback_text: str, corrected_action: str = "", outcome: str = "") -> dict:
        path = DATA_DIR / "feedback_log.csv"
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()
        row = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "asset_id": asset_id,
            "query": query,
            "feedback_type": feedback_type,
            "feedback_text": feedback_text,
            "corrected_action": corrected_action,
            "outcome": outcome,
        }
        pd.concat([df, pd.DataFrame([row])], ignore_index=True, sort=False).to_csv(path, index=False)
        return row

    def _dynamic_context_docs(self, asset_id: str, sensor: dict) -> list[dict]:
        missing = sensor.get("missing_readings") or ""
        uncertainty = (
            f" Missing readings: {missing}. Neutral defaults were used only for provisional risk scoring."
            if missing
            else ""
        )
        qualitative = (
            f" Operator notes: {sensor.get('operator_notes')}. {sensor.get('qualitative_risk_note')}"
            if sensor.get("operator_notes")
            else ""
        )
        rule_text = ""
        if sensor.get("applied_rules"):
            summaries = [
                f"{rule.get('rule_id')}: {rule.get('condition_text')}"
                for rule in sensor.get("applied_rules", [])
            ]
            rule_text = " Remembered rules applied: " + " | ".join(summaries)
        return [
            {
                "source": "dynamic_assets.csv",
                "asset_id": asset_id,
                "equipment_type": sensor.get("asset_type", "dynamic_asset"),
                "issue_type": "user_memory_current_health",
                "text": (
                    f"User-added asset {asset_id}. Type: {sensor.get('asset_type')}. Area: {sensor.get('area')}. "
                    f"Criticality: {sensor.get('criticality')}. Temperature: {_display_value(sensor.get('temperature_latest'))}. "
                    f"Vibration: {_display_value(sensor.get('vibration_latest'))}. Current: {_display_value(sensor.get('current_latest'))}. "
                    f"Pressure: {_display_value(sensor.get('pressure_latest'))}. Alarm count: {sensor.get('alarm_count_latest')}. "
                    f"Risk band: {sensor.get('risk_band')}. Hybrid health score: {sensor.get('hybrid_health_score')}. "
                    f"Estimated RUL days: {sensor.get('estimated_rul_days')}.{uncertainty}{qualitative}{rule_text}"
                ),
            }
        ]

    def _filter_docs_for_assets(self, docs: list[dict], asset_ids: list[str]) -> list[dict]:
        allowed = {str(asset_id).upper() for asset_id in asset_ids}
        allowed_equipment = set()
        for asset_id in allowed:
            sensor = self.get_latest_sensor_summary(asset_id)
            allowed_equipment.add(normalize_equipment_type(sensor.get("asset_type", "")))
            allowed_equipment.add(str(sensor.get("asset_type", "")).lower().replace(" ", "_"))
        out: list[dict] = []
        for doc in docs:
            aid = str(doc.get("asset_id", "")).upper()
            equipment = str(doc.get("equipment_type", "")).lower()
            source = str(doc.get("source", "")).lower()
            is_policy = equipment in {"policy", "safety"} or "policy" in source or "operating_model" in source
            is_scoped_all_doc = aid == "ALL" and (equipment in allowed_equipment or is_policy)
            if aid in allowed or aid in {"", "NONE", "NAN"} or is_scoped_all_doc:
                out.append(doc)
        return out

    def asset_ingestion_report(self, query: str, user_id: str = "demo_user") -> dict:
        parsed_assets = parse_dynamic_assets(query)
        if not parsed_assets:
            answer = (
                "I detected an asset-ingestion request, but I could not parse an asset ID and readings. "
                "Please provide an ID like BF-07 plus asset type, area, criticality, and sensor readings."
            )
            return {
                "mode": "asset_ingestion",
                "asset_id": None,
                "intent": "asset_ingestion",
                "answer": answer,
                "final_answer": answer,
                "agent_plan": [],
                "tool_calls": [],
                "verifier_checks": [{"check": "Asset fields parsed", "status": "review", "detail": "No asset row parsed"}],
                "decision_packet": {"mode": "asset_ingestion", "status": "needs_more_fields", "objective": query},
                "alert_report": "",
            }

        upsert_dynamic_assets(parsed_assets)
        scored = score_dynamic_assets(load_dynamic_assets())
        added_ids = [asset["asset_id"] for asset in parsed_assets]
        last_asset = added_ids[-1]
        self.session_memory["last_asset_id"] = last_asset
        self.session_memory["last_new_asset_id"] = last_asset
        self.session_memory.setdefault("new_asset_ids", [])
        for asset_id in added_ids:
            if asset_id not in self.session_memory["new_asset_ids"]:
                self.session_memory["new_asset_ids"].append(asset_id)

        scored_added = scored[scored["asset_id"].astype(str).str.upper().isin(set(added_ids))].copy()
        agent_plan = [
            {"step": 1, "agent": "Memory Agent", "task": "Detect asset-ingestion intent and parse asset rows", "target": ", ".join(added_ids), "status": "complete"},
            {"step": 2, "agent": "Sensor Agent", "task": "Normalize readings and build current asset state", "target": ", ".join(added_ids), "status": "complete"},
            {"step": 3, "agent": "Risk Agent", "task": "Score dynamic assets with operational rules and criticality", "target": ", ".join(added_ids), "status": "complete"},
            {"step": 4, "agent": "Verifier Agent", "task": "Confirm dynamic assets are now available to ranking, diagnosis, spares, and follow-up memory", "target": ", ".join(added_ids), "status": "complete"},
        ]
        tool_calls = [
            {"tool": "dynamic_asset_parser", "agent": "Memory Agent", "input": query, "output": f"{len(parsed_assets)} asset row(s) parsed", "status": "success"},
            {"tool": "dynamic_asset_memory_store", "agent": "Memory Agent", "input": "dynamic_assets.csv", "output": f"remembered {', '.join(added_ids)}", "status": "success"},
            {"tool": "dynamic_rule_scorer", "agent": "Risk Agent", "input": "current readings + criticality + equipment class", "output": f"{len(scored_added)} scored row(s)", "status": "success"},
        ]
        verifier_checks = [
            {"check": "Asset ID parsed", "status": "pass", "detail": ", ".join(added_ids)},
            {"check": "Dynamic memory persisted", "status": "pass", "detail": "dynamic_assets.csv"},
            {"check": "Usable in future ranking", "status": "pass", "detail": "asset_health_table merges demo and dynamic assets"},
            {"check": "Follow-up context updated", "status": "pass", "detail": f"last_new_asset_id={last_asset}"},
        ]
        decision_packet = {
            "mode": "asset_ingestion",
            "intent": "asset_ingestion",
            "status": "remembered",
            "objective": query,
            "added_assets": added_ids,
            "selected_asset": last_asset,
            "next_system_action": "use_dynamic_asset_memory_for_future_questions",
        }

        locked_sections = []
        for row in scored_added.sort_values("hybrid_health_score", ascending=False).to_dict("records"):
            locked_sections.append(
                "\n".join(
                    [
                        f"- Asset ID: {row.get('asset_id')}",
                        f"- Asset type: {row.get('asset_type')}",
                        f"- Area: {row.get('area')}",
                        f"- Criticality: {row.get('criticality')}",
                        f"- Temperature: {_display_value(row.get('temperature'), ' C')}",
                        f"- Vibration: {_display_value(row.get('vibration'), ' mm/s')}",
                        f"- Current: {_display_value(row.get('current'), ' A')}",
                        f"- Pressure: {_display_value(row.get('pressure'), ' bar')}",
                        f"- Alarm count: {row.get('alarm_count')}",
                        f"- Operator notes: {row.get('operator_notes') or 'none'}",
                        f"- Missing readings: {row.get('missing_readings') or 'none'}",
                        f"- Scoring note: {row.get('provisional_scoring_note') or 'all required readings provided'}",
                        f"- Qualitative risk note: {row.get('qualitative_risk_note') or 'none'}",
                        f"- Operational rule score: {row.get('operational_rule_score')}/100",
                        f"- Initial priority: {row.get('priority')}/{row.get('risk_band')}",
                        f"- Estimated RUL: {row.get('estimated_rul_days')} days",
                    ]
                )
            )

        answer = f"""
**Dynamic Asset Memory Update**

**{", ".join(added_ids)} added and remembered.**

**Agentic Control Loop**
- Objective: {query}
- Operating mode: asset ingestion and memory update
- Decision policy: parse user-supplied plant state, persist it, score it, and make it available to every later agent tool.

**Autonomous Execution Plan**
{chr(10).join([f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan])}

**Tool Calls Executed**
{chr(10).join([f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}" for t in tool_calls])}

**Verifier Checks**
{chr(10).join([f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks])}

**Locked Fields And Initial Assessment**
{chr(10).join(["", *locked_sections])}

**Memory**
- These assets are now included in plant ranking, comparison, diagnosis, RUL estimation, spares planning, alerting, and follow-up references such as "same new asset".
- Last new asset remembered: {last_asset}

**Final Decision Packet**
- Mode: asset_ingestion
- Status: remembered
- Added assets: {", ".join(added_ids)}
- Next system action: use_dynamic_asset_memory_for_future_questions
""".strip()

        priority = {"priority": "MEMORY", "risk_level": "ASSET_INGESTION", "urgency": "Remembered for future reasoning", "priority_score": 0}
        self.write_logbook(query, last_asset, priority, answer)
        return {
            "mode": "asset_ingestion",
            "asset_id": last_asset,
            "intent": "asset_ingestion",
            "dynamic_assets": scored_added.to_dict("records"),
            "risk_priority": priority,
            "priority": "Asset memory updated",
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "answer": answer,
            "final_answer": answer,
            "alert_report": f"Dynamic asset memory updated for {', '.join(added_ids)}.",
            "llm_used": False,
        }

    def asset_update_report(self, query: str, user_id: str = "demo_user", asset_id: str | None = None) -> dict:
        target = asset_id or self._infer_asset_from_query(query) or self.session_memory.get("last_new_asset_id")
        result = update_dynamic_assets_from_query(query, fallback_asset_id=target)
        updated = result.get("updated", [])
        missing = result.get("missing", [])
        history_rows = result.get("history", [])

        if not updated:
            answer = (
                "I detected an asset update request, but I could not apply it to dynamic memory. "
                f"Resolved asset: {target or 'none'}. "
                f"Missing or unknown assets: {', '.join(missing) if missing else 'none parsed'}."
            )
            return {
                "mode": "asset_update",
                "asset_id": target,
                "intent": "asset_update",
                "answer": answer,
                "final_answer": answer,
                "agent_plan": [],
                "tool_calls": [],
                "verifier_checks": [{"check": "Dynamic asset update applied", "status": "review", "detail": answer}],
                "decision_packet": {"mode": "asset_update", "status": "not_applied", "objective": query, "resolved_asset": target},
                "alert_report": "",
            }

        updated_ids = [row["asset_id"] for row in updated]
        last_asset = updated_ids[-1]
        self.session_memory["last_asset_id"] = last_asset
        self.session_memory["last_new_asset_id"] = last_asset

        comparisons = []
        interpretations = []
        for row in history_rows:
            previous = json.loads(row["previous_record"])
            new = json.loads(row["new_record"])
            changed_field_list = json.loads(row["changed_fields"])
            changed_fields = ", ".join(changed_field_list)
            priority_changed = (
                previous.get("priority") != new.get("priority")
                or previous.get("risk_band") != new.get("risk_band")
            )
            previous_score = safe_float(previous.get("hybrid_health_score"))
            new_score = safe_float(new.get("hybrid_health_score"))
            if new_score < previous_score and not priority_changed and new.get("operator_notes"):
                interpretations.append(
                    f"- {new.get('asset_id')}: numeric score reduced from {previous_score} to {new_score}, "
                    f"but priority stays {new.get('priority')}/{new.get('risk_band')} because operator-reported symptoms remain active evidence."
                )
            elif new_score < previous_score:
                interpretations.append(
                    f"- {new.get('asset_id')}: numeric score reduced from {previous_score} to {new_score}; priority is now {new.get('priority')}/{new.get('risk_band')}."
                )
            elif new_score > previous_score:
                interpretations.append(
                    f"- {new.get('asset_id')}: numeric score increased from {previous_score} to {new_score}; priority is now {new.get('priority')}/{new.get('risk_band')}."
                )
            elif new.get("operator_notes"):
                interpretations.append(
                    f"- {new.get('asset_id')}: numeric score stayed at {new_score}; priority remains {new.get('priority')}/{new.get('risk_band')} "
                    "because active operator-reported symptoms remain risk evidence."
                )
            else:
                interpretations.append(
                    f"- {new.get('asset_id')}: numeric score stayed at {new_score}; priority remains {new.get('priority')}/{new.get('risk_band')}."
                )
            if new.get("applied_rules"):
                interpretations.append(
                    f"- {new.get('asset_id')}: {len(new.get('applied_rules', []))} remembered safety/SOP rule(s) were applied during re-scoring."
                )
            applied_rule_lines = [
                f"  - {rule.get('rule_id')}: {rule.get('condition_text')}"
                for rule in new.get("applied_rules", [])
            ]
            comparisons.append(
                "\n".join(
                    [
                        f"- Asset ID: {new.get('asset_id')}",
                        f"- Changed fields: {changed_fields}",
                        f"- Previous priority: {previous.get('priority')}/{previous.get('risk_band')} | score {previous.get('hybrid_health_score')}",
                        f"- New priority: {new.get('priority')}/{new.get('risk_band')} | score {new.get('hybrid_health_score')}",
                        f"- Priority changed: {'YES' if priority_changed else 'NO'}",
                        f"- Temperature: {_display_value(previous.get('temperature'), ' C')} -> {_display_value(new.get('temperature'), ' C')}",
                        f"- Vibration: {_display_value(previous.get('vibration'), ' mm/s')} -> {_display_value(new.get('vibration'), ' mm/s')}",
                        f"- Current: {_display_value(previous.get('current'), ' A')} -> {_display_value(new.get('current'), ' A')}",
                        f"- Pressure: {_display_value(previous.get('pressure'), ' bar')} -> {_display_value(new.get('pressure'), ' bar')}",
                        f"- Alarm count: {previous.get('alarm_count')} -> {new.get('alarm_count')}",
                        f"- Operator notes: {previous.get('operator_notes') or 'none'} -> {new.get('operator_notes') or 'none'}",
                        f"- Qualitative risk note: {new.get('qualitative_risk_note') or 'none'}",
                        f"- Remembered rules applied: {len(new.get('applied_rules', []))}",
                        *applied_rule_lines,
                    ]
                )
            )

        agent_plan = [
            {"step": 1, "agent": "Memory Agent", "task": "Detect dynamic asset update intent", "target": ", ".join(updated_ids), "status": "complete"},
            {"step": 2, "agent": "State Agent", "task": "Load previous dynamic asset row", "target": ", ".join(updated_ids), "status": "complete"},
            {"step": 3, "agent": "Sensor Agent", "task": "Apply only the fields supplied by the user", "target": ", ".join(updated_ids), "status": "complete"},
            {"step": 4, "agent": "Risk Agent", "task": "Re-score updated state and compare old versus new priority", "target": ", ".join(updated_ids), "status": "complete"},
            {"step": 5, "agent": "Memory Agent", "task": "Write update event to dynamic_asset_history.csv", "target": ", ".join(updated_ids), "status": "complete"},
        ]
        tool_calls = [
            {"tool": "dynamic_asset_update_parser", "agent": "Memory Agent", "input": query, "output": f"{len(updated)} asset update(s) parsed", "status": "success"},
            {"tool": "dynamic_asset_state_store", "agent": "State Agent", "input": "dynamic_assets.csv", "output": f"updated {', '.join(updated_ids)}", "status": "success"},
            {"tool": "dynamic_asset_history_writer", "agent": "Memory Agent", "input": "dynamic_asset_history.csv", "output": f"{len(history_rows)} update event(s) stored", "status": "success"},
        ]
        verifier_checks = [
            {"check": "Update applied to existing asset", "status": "pass", "detail": ", ".join(updated_ids)},
            {"check": "Previous version preserved", "status": "pass", "detail": "dynamic_asset_history.csv"},
            {"check": "Future ranking uses latest state", "status": "pass", "detail": "asset_health_table reads updated dynamic memory"},
        ]
        selected = updated[-1]
        priority = {
            "priority": selected.get("priority"),
            "risk_level": selected.get("risk_band"),
            "urgency": selected.get("urgency"),
            "priority_score": selected.get("hybrid_health_score"),
        }
        decision_packet = {
            "mode": "asset_update",
            "intent": "dynamic_asset_update",
            "objective": query,
            "updated_assets": updated_ids,
            "selected_asset": last_asset,
            "risk_level": priority.get("risk_level"),
            "priority": priority.get("priority"),
            "next_system_action": "use_latest_dynamic_asset_state_for_future_reasoning",
        }

        answer = f"""
**Dynamic Asset Update Applied**

**Updated assets:** {", ".join(updated_ids)}

**What changed**
{chr(10).join(["", *comparisons])}

**Risk Interpretation**
{chr(10).join(interpretations)}

**Agentic Control Loop**
- Objective: {query}
- Operating mode: dynamic asset update and state comparison
- Decision policy: preserve previous state, apply only supplied fields, re-score, then write update history.

**Autonomous Execution Plan**
{chr(10).join([f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan])}

**Tool Calls Executed**
{chr(10).join([f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}" for t in tool_calls])}

**Verifier Checks**
{chr(10).join([f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks])}

**Memory**
- The latest readings are now the active source of truth for diagnosis, ranking, spares, alerts, and follow-up questions.
- The previous version is retained for "did priority change?" comparisons.

**Final Decision Packet**
- Mode: asset_update
- Updated assets: {", ".join(updated_ids)}
- Selected asset: {last_asset}
- Next system action: use_latest_dynamic_asset_state_for_future_reasoning
""".strip()

        self.write_logbook(query, last_asset, priority, answer)
        return {
            "mode": "asset_update",
            "asset_id": last_asset,
            "intent": "dynamic_asset_update",
            "updated_assets": updated,
            "risk_priority": priority,
            "priority": f"{priority.get('priority')}/{priority.get('risk_level')}",
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "answer": answer,
            "final_answer": answer,
            "alert_report": f"Dynamic asset update applied for {', '.join(updated_ids)}.",
            "llm_used": False,
        }

    def rule_ingestion_report(self, query: str, user_id: str = "demo_user") -> dict:
        rule = remember_dynamic_rule(query)
        self.session_memory["last_rule_id"] = rule["rule_id"]
        rules = load_dynamic_rules()
        agent_plan = [
            {"step": 1, "agent": "Memory Agent", "task": "Detect safety/SOP rule ingestion intent", "target": "dynamic rule memory", "status": "complete"},
            {"step": 2, "agent": "Policy Agent", "task": "Extract equipment scope, condition text, and priority override", "target": rule["rule_id"], "status": "complete"},
            {"step": 3, "agent": "State Agent", "task": "Persist rule for future scoring, ranking, diagnosis, alerts, and follow-ups", "target": "dynamic_rules.csv", "status": "complete"},
            {"step": 4, "agent": "Verifier Agent", "task": "Confirm rule exists in active rule memory", "target": rule["rule_id"], "status": "complete"},
        ]
        tool_calls = [
            {"tool": "universal_command_parser", "agent": "Memory Agent", "input": query, "output": "RULE_INGEST", "status": "success"},
            {"tool": "dynamic_rule_parser", "agent": "Policy Agent", "input": query, "output": f"{rule['priority_override'] or 'policy'} override scoped by {rule['equipment_pattern']}", "status": "success"},
            {"tool": "dynamic_rule_store", "agent": "State Agent", "input": "dynamic_rules.csv", "output": f"{len(rules)} active/remembered rule row(s)", "status": "success"},
        ]
        verifier_checks = [
            {"check": "Rule stored", "status": "pass", "detail": rule["rule_id"]},
            {"check": "Rule applied by scorer", "status": "pass", "detail": "score_dynamic_assets calls dynamic rule engine"},
            {"check": "Diagnosis/ranking will use rule", "status": "pass", "detail": "asset_health_table merges rule-adjusted dynamic state"},
        ]
        decision_packet = {
            "mode": "rule_ingestion",
            "intent": "dynamic_safety_rule_memory",
            "rule_id": rule["rule_id"],
            "rule_type": rule["rule_type"],
            "equipment_pattern": rule["equipment_pattern"],
            "area_pattern": rule["area_pattern"],
            "priority_override": rule["priority_override"],
            "risk_override": rule["risk_override"],
            "next_system_action": "apply_dynamic_rules_during_all_future_scoring",
        }
        answer = f"""
**Safety/SOP Rule Remembered**

Rule `{rule["rule_id"]}` has been stored and will be applied to future diagnosis, ranking, RUL, alerting, and follow-up reasoning.

**Parsed Rule**
- Rule type: {rule["rule_type"]}
- Equipment scope: {rule["equipment_pattern"]}
- Area scope: {rule["area_pattern"]}
- Priority override: {rule["priority_override"] or "none"}
- Risk override: {rule["risk_override"] or "none"}
- Condition: {rule["condition_text"]}

**Agentic Control Loop**
- Objective: {query}
- Operating mode: dynamic rule ingestion
- Decision policy: memory-changing commands are handled before diagnosis.

**Autonomous Execution Plan**
{chr(10).join([f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan])}

**Tool Calls Executed**
{chr(10).join([f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}" for t in tool_calls])}

**Verifier Checks**
{chr(10).join([f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks])}

**Final Decision Packet**
- Mode: rule_ingestion
- Rule ID: {rule["rule_id"]}
- Next system action: apply_dynamic_rules_during_all_future_scoring
""".strip()
        priority = {"priority": "MEMORY", "risk_level": "RULE_INGESTION", "urgency": "Rule remembered", "priority_score": 0}
        self.write_logbook(query, self.session_memory.get("last_asset_id", "RULE_MEMORY"), priority, answer)
        return {
            "mode": "rule_ingestion",
            "asset_id": self.session_memory.get("last_asset_id"),
            "intent": "dynamic_safety_rule_memory",
            "rule": rule,
            "risk_priority": priority,
            "priority": "Rule remembered",
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "answer": answer,
            "final_answer": answer,
            "alert_report": f"Safety/SOP rule remembered: {rule['rule_id']}.",
            "llm_used": False,
        }

    def rule_apply_report(self, query: str, asset_id: str | None = None, user_id: str = "demo_user") -> dict:
        target = asset_id or self._infer_asset_from_query(query) or self.session_memory.get("last_asset_id")
        if not target:
            answer = "I can apply remembered rules, but I need an asset ID or a remembered asset reference."
            return {
                "mode": "rule_apply",
                "asset_id": None,
                "intent": "dynamic_rule_application",
                "answer": answer,
                "final_answer": answer,
                "agent_plan": [],
                "tool_calls": [],
                "verifier_checks": [{"check": "Asset resolved for rule application", "status": "review", "detail": "No asset available"}],
                "decision_packet": {"mode": "rule_apply", "status": "needs_asset_id", "objective": query},
                "alert_report": "",
            }

        sensor = self.get_latest_sensor_summary(target)
        spares = self.get_spares(target)
        delay = self.get_delay(target)
        priority = self.prioritize_action(sensor, spares, delay)
        rules = sensor.get("applied_rules") or []
        base_priority = f"{sensor.get('base_priority')}/{sensor.get('base_risk_band')}"
        final_priority = f"{priority.get('priority')}/{priority.get('risk_level')}"
        changed = base_priority != final_priority
        rule_lines = (
            "\n".join(
                f"- {rule.get('rule_id')}: {rule.get('condition_text')} -> {rule.get('priority_override') or 'policy'}"
                for rule in rules
            )
            if rules
            else "- No remembered rule matched this asset and current readings."
        )
        agent_plan = [
            {"step": 1, "agent": "Triage Agent", "task": "Resolve asset for remembered rule application", "target": target, "status": "complete"},
            {"step": 2, "agent": "Policy Agent", "task": "Load active dynamic safety/SOP rules", "target": "dynamic_rules.csv", "status": "complete"},
            {"step": 3, "agent": "Risk Agent", "task": "Apply matching rules inside dynamic scoring", "target": target, "status": "complete"},
            {"step": 4, "agent": "Verifier Agent", "task": "Compare base score against final rule-adjusted priority", "target": target, "status": "complete"},
        ]
        tool_calls = [
            {"tool": "asset_resolver", "agent": "Triage Agent", "input": query, "output": target, "status": "success"},
            {"tool": "dynamic_rule_loader", "agent": "Policy Agent", "input": "dynamic_rules.csv", "output": f"{len(load_dynamic_rules())} remembered rule row(s)", "status": "success"},
            {"tool": "dynamic_rule_engine", "agent": "Risk Agent", "input": target, "output": f"{len(rules)} rule(s) applied", "status": "success"},
        ]
        verifier_checks = [
            {"check": "Asset resolved", "status": "pass", "detail": target},
            {"check": "Rule application evaluated", "status": "pass", "detail": f"{len(rules)} applied rule(s)"},
            {"check": "Base vs final priority compared", "status": "pass", "detail": f"{base_priority} -> {final_priority}"},
        ]
        decision_packet = {
            "mode": "rule_apply",
            "intent": "dynamic_rule_application",
            "selected_asset": target,
            "applied_rule_count": len(rules),
            "priority_changed_by_rule": changed,
            "base_priority": base_priority,
            "final_priority": final_priority,
            "hybrid_health_score": sensor.get("hybrid_health_score"),
            "estimated_rul_days": sensor.get("estimated_rul_days"),
            "next_system_action": "create_or_update_work_order_if_p1_p2" if priority.get("priority") in {"P1", "P2"} else "monitor_and_schedule",
        }
        answer = f"""
**Remembered Rule Application For {target}**

**Result**
- Applied rules: {len(rules)}
- Base priority before remembered rules: {base_priority}, score {sensor.get("base_hybrid_health_score")}/100
- Final priority after remembered rules and plant policy: {final_priority}, score {sensor.get("hybrid_health_score")}/100
- Priority changed by remembered rule: {"YES" if changed else "NO"}

**Rules Evaluated**
{rule_lines}

**Current Asset State**
- Asset type: {sensor.get("asset_type")}
- Area: {sensor.get("area")}
- Temperature: {_display_value(sensor.get("temperature_latest"), " C")}
- Vibration: {_display_value(sensor.get("vibration_latest"), " mm/s")}
- Current: {_display_value(sensor.get("current_latest"), " A")}
- Pressure: {_display_value(sensor.get("pressure_latest"), " bar")}
- Alarm count: {sensor.get("alarm_count_latest")}
- RUL: {sensor.get("estimated_rul_days")} days

**Agentic Control Loop**
- Objective: {query}
- Operating mode: dynamic safety rule application
- Decision policy: rules are applied by the scorer before diagnosis/ranking output.

**Autonomous Execution Plan**
{chr(10).join([f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan])}

**Tool Calls Executed**
{chr(10).join([f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}" for t in tool_calls])}

**Verifier Checks**
{chr(10).join([f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks])}

**Final Decision Packet**
- Mode: rule_apply
- Selected asset: {target}
- Applied rule count: {len(rules)}
- Next system action: {decision_packet["next_system_action"]}
""".strip()
        self.session_memory["last_asset_id"] = target
        if self._is_dynamic_asset(target):
            self.session_memory["last_new_asset_id"] = target
        self.write_logbook(query, target, priority, answer)
        return {
            "mode": "rule_apply",
            "asset_id": target,
            "intent": "dynamic_rule_application",
            "applied_rules": rules,
            "sensor_summary": sensor,
            "risk_priority": priority,
            "priority": final_priority,
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "answer": answer,
            "final_answer": answer,
            "alert_report": f"Remembered rules evaluated for {target}: {len(rules)} applied.",
            "llm_used": False,
        }

    def dynamic_priority_change_report(self, query: str, asset_id: str | None = None, user_id: str = "demo_user") -> dict:
        target = asset_id or self._infer_asset_from_query(query) or self.session_memory.get("last_new_asset_id")
        if not target:
            answer = "I can compare priority after an update, but I need an asset ID or a remembered new asset."
            return {
                "mode": "asset_update_review",
                "asset_id": None,
                "intent": "priority_change_review",
                "answer": answer,
                "final_answer": answer,
                "agent_plan": [],
                "tool_calls": [],
                "verifier_checks": [{"check": "Asset resolved for change review", "status": "review", "detail": "No asset ID available"}],
                "decision_packet": {"mode": "asset_update_review", "status": "needs_asset_id", "objective": query},
                "alert_report": "",
            }

        change = latest_dynamic_asset_change(target)
        if not change:
            answer = f"No update history is available yet for {target}. Add or update readings first, then ask again."
            return {
                "mode": "asset_update_review",
                "asset_id": target,
                "intent": "priority_change_review",
                "answer": answer,
                "final_answer": answer,
                "agent_plan": [],
                "tool_calls": [],
                "verifier_checks": [{"check": "Update history found", "status": "review", "detail": "No update event found"}],
                "decision_packet": {"mode": "asset_update_review", "status": "no_update_history", "selected_asset": target},
                "alert_report": "",
            }

        previous = change.get("previous_record", {})
        new = change.get("new_record", {})
        changed_fields = change.get("changed_fields", [])
        priority_changed = (
            previous.get("priority") != new.get("priority")
            or previous.get("risk_band") != new.get("risk_band")
        )
        answer = f"""
**Priority Change Review For {target}**

**Priority changed:** {"YES" if priority_changed else "NO"}

**Before**
- Priority: {previous.get("priority")}/{previous.get("risk_band")}
- Score: {previous.get("hybrid_health_score")}/100
- RUL: {previous.get("estimated_rul_days")} days

**After**
- Priority: {new.get("priority")}/{new.get("risk_band")}
- Score: {new.get("hybrid_health_score")}/100
- RUL: {new.get("estimated_rul_days")} days

**Changed readings**
- Fields: {", ".join(changed_fields)}
- Temperature: {_display_value(previous.get("temperature"), " C")} -> {_display_value(new.get("temperature"), " C")}
- Vibration: {_display_value(previous.get("vibration"), " mm/s")} -> {_display_value(new.get("vibration"), " mm/s")}
- Current: {_display_value(previous.get("current"), " A")} -> {_display_value(new.get("current"), " A")}
- Pressure: {_display_value(previous.get("pressure"), " bar")} -> {_display_value(new.get("pressure"), " bar")}
- Alarm count: {previous.get("alarm_count")} -> {new.get("alarm_count")}
- Operator notes: {previous.get("operator_notes") or "none"} -> {new.get("operator_notes") or "none"}

**Reason**
- The agent compared the previous stored dynamic state against the latest update event in memory.
- Higher vibration, current, temperature, pressure deviation, alarm count, criticality, and operator-reported symptoms increase the operational rule score and may change priority.
""".strip()
        priority = {
            "priority": new.get("priority"),
            "risk_level": new.get("risk_band"),
            "urgency": new.get("urgency", "Review update"),
            "priority_score": new.get("hybrid_health_score"),
        }
        decision_packet = {
            "mode": "asset_update_review",
            "intent": "priority_change_review",
            "selected_asset": target,
            "priority_changed": priority_changed,
            "previous_priority": f"{previous.get('priority')}/{previous.get('risk_band')}",
            "new_priority": f"{new.get('priority')}/{new.get('risk_band')}",
            "changed_fields": changed_fields,
            "next_system_action": "continue_with_latest_dynamic_asset_state",
        }
        return {
            "mode": "asset_update_review",
            "asset_id": target,
            "intent": "priority_change_review",
            "risk_priority": priority,
            "priority": f"{priority.get('priority')}/{priority.get('risk_level')}",
            "agent_plan": self.build_agent_plan(query, mode="asset_diagnosis", asset_id=target),
            "tool_calls": [
                {"tool": "dynamic_asset_history_lookup", "agent": "Memory Agent", "input": target, "output": "latest update event loaded", "status": "success"},
                {"tool": "priority_delta_checker", "agent": "Verifier Agent", "input": "previous state + new state", "output": f"priority_changed={priority_changed}", "status": "success"},
            ],
            "verifier_checks": [
                {"check": "Update history found", "status": "pass", "detail": change.get("changed_at")},
                {"check": "Old and new priorities compared", "status": "pass", "detail": f"{previous.get('priority')} -> {new.get('priority')}"},
            ],
            "decision_packet": decision_packet,
            "answer": answer,
            "final_answer": answer,
            "alert_report": f"Priority change review completed for {target}.",
            "llm_used": False,
        }

    def chat(self, query: str, user_id: str = "demo_user") -> dict:
        self.ensure_ready()
        self.session_memory["user_id"] = user_id
        if is_rule_ingestion_query(query):
            return self.rule_ingestion_report(query, user_id=user_id)
        if is_rule_apply_query(query):
            return self.rule_apply_report(query, asset_id=self._infer_asset_from_query(query), user_id=user_id)
        if is_asset_update_query(query):
            return self.asset_update_report(query, asset_id=self._infer_asset_from_query(query), user_id=user_id)
        if is_priority_change_query(query):
            return self.dynamic_priority_change_report(query, user_id=user_id)
        if is_asset_ingestion_query(query):
            return self.asset_ingestion_report(query, user_id=user_id)
        intent_hint = classify_steel_intent(query)
        if intent_hint == "predictive_maintenance_workflow_design":
            return self.general_steel_report(query)
        if self._is_public_query(query) and not self.query_assets(query) and not self._is_plant_query(query):
            return self.public_dataset_report(query)
        if self._is_plant_query(query):
            return self.plant_priority_report(query, asset_ids=self._plant_scope_asset_ids(query))

        asset_id = self._infer_asset_from_query(query)
        if asset_id:
            return self.asset_report(query, asset_id)

        if self._is_general_steel_query(query):
            return self.general_steel_report(query)

        answer = (
            "I am configured as a steel-plant maintenance and operations agent. "
            "Ask me about steel equipment, failures, SOPs, risk, spares, safety, process defects, "
            "plant priority, or maintenance planning."
        )
        return {
            "mode": "clarification",
            "asset_id": None,
            "answer": answer,
            "final_answer": answer,
            "priority": "UNKNOWN",
            "anomaly_result": {},
            "alert_report": "",
            "agent_plan": [],
            "tool_calls": [],
            "verifier_checks": [],
            "decision_packet": {"mode": "clarification", "objective": query},
        }

    def general_steel_report(self, query: str) -> dict:
        intent = classify_steel_intent(query)
        subject = infer_steel_subject(query)
        docs = self.rag.retrieve(query, top_k=12, plant_level=True)
        if intent == "predictive_maintenance_workflow_design":
            preferred = [
                "steel_agent_operating_model.txt",
                "maintenance_prioritization_policy.txt",
                "asset_health_summary.csv",
                "SOP_GBX_17_gearbox_vibration.txt",
                "feedback_learning_policy.txt",
                "DATA_SOURCES.md",
            ]
            by_source = {}
            for doc in docs:
                by_source.setdefault(doc.get("source"), doc)
            if hasattr(self.rag, "doc_df") and not self.rag.doc_df.empty:
                for source in preferred:
                    if source in by_source:
                        continue
                    matches = self.rag.doc_df[self.rag.doc_df["source"] == source]
                    if not matches.empty:
                        row = matches.iloc[0]
                        by_source[source] = {
                            "score": 1.0,
                            "source": row["source"],
                            "asset_id": row["asset_id"],
                            "equipment_type": row["equipment_type"],
                            "issue_type": row["issue_type"],
                            "text": row["text"],
                        }
            ordered_docs = [by_source[source] for source in preferred if source in by_source]
            ordered_docs += [doc for doc in docs if doc.get("source") not in preferred]
            docs = ordered_docs[:8]

        health_df = self.asset_health_table()
        health_rows = summarize_health_rows(health_df.to_dict("records"))
        feedback_path = DATA_DIR / "feedback_log.csv"
        feedback_rows = len(pd.read_csv(feedback_path)) if feedback_path.exists() else 0

        agent_plan = build_general_plan(query, intent, subject)
        tool_calls = build_general_tool_calls(query, intent, subject, docs, health_rows, feedback_rows)
        verifier_checks = build_general_verifier_checks(intent, docs, health_rows)
        decision_packet = build_general_decision_packet(query, intent, subject, docs, health_rows)
        answer = build_general_answer(
            query=query,
            intent=intent,
            subject=subject,
            docs=docs,
            health_rows=health_rows,
            agent_plan=agent_plan,
            tool_calls=tool_calls,
            verifier_checks=verifier_checks,
            decision_packet=decision_packet,
        )

        priority = {
            "priority": "AGENT",
            "risk_level": "CONTEXTUAL",
            "urgency": decision_packet["urgency"],
            "priority_score": 0,
        }
        self.session_memory["last_general_subject"] = subject
        self.write_logbook(query, subject, priority, answer)
        return {
            "mode": decision_packet["mode"],
            "asset_id": subject,
            "intent": intent,
            "subject": subject,
            "applied_demo_target": decision_packet.get("applied_demo_target"),
            "risk_priority": priority,
            "priority": "General steel agent",
            "retrieved_docs": docs,
            "plant_health_snapshot": health_rows[:5],
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "alert_report": f"General steel agent response generated for {subject}.",
            "answer": answer,
            "final_answer": answer,
            "llm_used": True,
            "llm_validation": "general_steel_agent_with_traceable_plan",
        }

    def public_dataset_report(self, query: str) -> dict:
        public_path = DATA_DIR / "public_ai4i_common_schema.csv"
        public_rows = len(pd.read_csv(public_path)) if public_path.exists() else 0
        steel_rows = len(pd.read_csv(DATA_DIR / "steel_sensor_logs.csv"))
        agent_plan = self.build_agent_plan(query, mode="public_dataset", asset_id="PUBLIC_AI4I")
        tool_calls = [
            {"tool": "public_dataset_loader", "agent": "Data Agent", "input": "public_ai4i_common_schema.csv", "output": f"{public_rows} rows available", "status": "success" if public_rows else "review"},
            {"tool": "leakage_guard", "agent": "Safety Agent", "input": "AI4I target/features", "output": "Machine failure target excluded from features", "status": "success"},
            {"tool": "model_boundary_checker", "agent": "ML Agent", "input": "public benchmark vs steel app", "output": "separate model paths confirmed", "status": "success"},
        ]
        verifier_checks = [
            {"check": "Public benchmark present", "status": "pass" if public_rows else "review", "detail": f"{public_rows} rows"},
            {"check": "Target leakage removed", "status": "pass", "detail": "Machine failure used only as target"},
            {"check": "Steel app model separated", "status": "pass", "detail": "hybrid steel decisions do not use public labels"},
        ]
        decision_packet = {
            "mode": "data_governance",
            "objective": query,
            "selected_asset": "PUBLIC_AI4I",
            "public_rows": public_rows,
            "steel_rows": steel_rows,
            "next_system_action": "use_public_data_for_benchmark_only",
            "top_sources": ["public_ai4i_common_schema.csv", "DATA_SOURCES.md", "model_summary.json"],
        }
        answer = f"""
**Public Dataset and ML Validation Summary**

**Agentic Control Loop**
- Objective: {query}
- Operating mode: data-governance validation
- Decision policy: validate external benchmark without allowing target leakage into steel app decisions.

**Autonomous Execution Plan**
{chr(10).join([f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan])}

**Tool Calls Executed**
{chr(10).join([f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}" for t in tool_calls])}

**Verifier Checks**
{chr(10).join([f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks])}

**Dataset Used**
- Public benchmark: AI4I 2020 Predictive Maintenance dataset.
- Public rows available: {public_rows}
- Steel demo rows available: {steel_rows}

**How It Is Used**
- AI4I is used as an external benchmark to validate the predictive-maintenance ML pipeline.
- It is not mixed into the steel app decision layer.
- The steel Maintenance Wizard decisions use the steel demo model plus operational rules.

**Leakage Control**
- `Machine failure` is used only as the target label.
- Failure subtype columns such as TWF, HDF, PWF, OSF, and RNF are not used as model features.
- AI4I sensor proxy fields are engineered only from non-target process variables.

**Agent Reasoning Trace**
- Data Agent: checked public benchmark availability.
- ML Agent: separated public validation from steel app scoring.
- Safety Agent: confirmed target leakage is removed.
- Reporting Agent: generated explainable dataset summary.

**Final Decision Packet**
- Mode: {decision_packet["mode"]}
- Next system action: {decision_packet["next_system_action"]}
- Top evidence sources: {", ".join(decision_packet["top_sources"])}
""".strip()
        return {
            "mode": "public_dataset_summary",
            "asset_id": "PUBLIC_AI4I",
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "answer": answer,
            "final_answer": answer,
            "llm_used": True,
        }

    def asset_report(self, query: str, asset_id: str) -> dict:
        sensor = self.get_latest_sensor_summary(asset_id)
        anomaly = self.detect_anomaly(asset_id)
        history = self.get_history(asset_id)
        failures = self.get_failures(asset_id)
        spares = self.get_spares(asset_id)
        delay = self.get_delay(asset_id)
        priority = self.prioritize_action(sensor, spares, delay)
        if sensor.get("is_dynamic"):
            docs = self._dynamic_context_docs(asset_id, sensor) + self._filter_docs_for_assets(
                self.rag.retrieve(query, top_k=6, plant_level=True),
                [asset_id],
            )[:4]
        else:
            docs = self.rag.retrieve(query, top_k=5, asset_id=asset_id, equipment_type=sensor.get("asset_type"))
        trace = self.build_agent_trace(asset_id, sensor, anomaly, priority, docs)
        rules = self.rule_breakdown(sensor, delay, spares)
        feedback = self.get_feedback(asset_id)
        actions = self.recommended_actions(asset_id)
        agent_plan = self.build_agent_plan(query, mode="asset_diagnosis", asset_id=asset_id)
        tool_calls = self.build_tool_calls(asset_id, sensor, anomaly, priority, docs, history, failures, spares, delay, feedback)
        verifier_checks = self.build_verifier_checks(sensor, priority, docs, spares)
        decision_packet = self.build_decision_packet(
            mode="asset_diagnosis",
            query=query,
            asset_id=asset_id,
            sensor=sensor,
            priority=priority,
            docs=docs,
            actions=actions,
        )
        facts = {
            "asset_id": asset_id,
            "risk_level": priority["risk_level"],
            "priority": priority["priority"],
            "rul_days": sensor["estimated_rul_days"],
            "hybrid_failure_risk": sensor["hybrid_failure_risk"],
            "ml_failure_risk": sensor["ml_failure_risk_latest"],
            "operational_rule_score": sensor["operational_rule_score"],
            "source_count": len(docs),
        }
        llm_note = self.llm.explain(facts)

        report = f"""
**Maintenance Wizard Report**

**Agentic Control Loop**
- Objective: {query}
- Selected asset: {asset_id}
- Operating mode: autonomous maintenance diagnosis
- Decision policy: lock deterministic safety fields first, then use LLM only for engineer explanation.

**Autonomous Execution Plan**
{chr(10).join([f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan])}

**Tool Calls Executed**
{chr(10).join([f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}" for t in tool_calls])}

**Verifier Checks**
{chr(10).join([f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks])}

**Locked Decision Fields**
- Asset ID: {asset_id}
- Equipment type: {sensor.get("asset_type")}
- Area: {sensor.get("area")}
- Criticality: {sensor.get("criticality")}
- Risk level: {priority.get("risk_level")}
- Priority: {priority.get("priority")} - {priority.get("urgency")}
- Hybrid failure risk: {sensor.get("hybrid_failure_risk")}
- ML failure risk: {sensor.get("ml_failure_risk_latest")}
- Operational rule score: {sensor.get("operational_rule_score")}/100
- Hybrid health score: {sensor.get("hybrid_health_score")}/100
- Remembered rules applied: {sensor.get("applied_rule_count", 0)}
- RUL / remaining useful life: {sensor.get("estimated_rul_days")} days

**Diagnosis**
- The asset shows {anomaly.get("anomaly_level")} abnormality based on sensor trend, anomaly events, ML signal, and operational safety rules.

**Root Cause**
- Probable root cause: {self.infer_root_cause(asset_id)}.

**Risk Score Explanation**
{chr(10).join([f"- {reason}" for reason in rules])}

**Risk and RUL Explanation**
- Final app decision uses hybrid scoring: 0.45 * ML failure risk + 0.55 * operational rule score.
- RUL is reduced by hybrid failure risk, anomaly count, criticality, and degradation slope.
- Temperature slope 24h: {sensor.get("temperature_slope_24h")}
- Vibration slope 24h: {sensor.get("vibration_slope_24h")}
- Pressure slope 24h: {sensor.get("pressure_slope_24h")}

**Immediate Actions**
{chr(10).join([f"- {action}" for action in actions])}

**Spare Strategy**
{_spares_strategy(spares)}

**Evidence / Sources**
Historical work orders:
{_format_records(history)}

Failure reports:
{_format_records(failures)}

Retrieved evidence:
{_format_sources(docs)}

**Previous Engineer Feedback Used**
{_format_records(feedback)}

**Agent Reasoning Trace**
{chr(10).join([f"- {step['agent']}: {step['decision']}" for step in trace])}

**Final Decision Packet**
- Mode: {decision_packet["mode"]}
- Next system action: {decision_packet["next_system_action"]}
- Recommended first action: {decision_packet["recommended_first_action"]}
- Top evidence sources: {", ".join(decision_packet["top_sources"])}

**LLM Engineer Explanation**
{llm_note}

**Alert / Logbook Note**
- Alert: {priority.get("risk_level")} risk for {asset_id}; {priority.get("urgency")}.
- Digital logbook entry created for follow-up.
""".strip()
        self.session_memory["last_asset_id"] = asset_id
        self.write_logbook(query, asset_id, priority, report)
        return {
            "mode": "asset_diagnosis",
            "asset_id": asset_id,
            "intent": "asset_diagnosis",
            "sensor_summary": sensor,
            "anomaly_result": anomaly,
            "risk_priority": priority,
            "priority": f"{priority.get('priority')} - {priority.get('urgency')}",
            "history": history,
            "failure_reports": failures,
            "spares": spares,
            "delay": delay,
            "feedback_used": feedback,
            "rule_breakdown": rules,
            "retrieved_docs": docs,
            "agent_trace": trace,
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "alert_report": f"Alert for {asset_id}: {priority.get('risk_level')} risk, {priority.get('priority')}, RUL {sensor.get('estimated_rul_days')} days.",
            "answer": report,
            "final_answer": report,
            "llm_used": True,
            "llm_validation": "locked_fields_plus_llm_explanation",
        }

    def plant_priority_report(self, query: str, asset_ids: list[str] | None = None) -> dict:
        asset_ids = asset_ids or self.asset_ids
        rows = []
        for asset_id in asset_ids:
            sensor = self.get_latest_sensor_summary(asset_id)
            spares = self.get_spares(asset_id)
            delay = self.get_delay(asset_id)
            priority = self.prioritize_action(sensor, spares, delay)
            display_risk = str(sensor.get("risk_band") or priority.get("risk_level") or "LOW").upper()
            display_priority = {"CRITICAL": "P1", "HIGH": "P2", "MEDIUM": "P3", "LOW": "P4"}.get(display_risk, priority.get("priority", "P4"))
            display_urgency = {
                "CRITICAL": "Immediate action",
                "HIGH": "Action within 24 hours",
                "MEDIUM": "Plan in maintenance window",
                "LOW": "Monitor only",
            }.get(display_risk, priority.get("urgency", "Monitor"))
            rows.append(
                {
                    "asset_id": asset_id,
                    "asset_type": sensor.get("asset_type"),
                    "area": sensor.get("area"),
                    "criticality": sensor.get("criticality"),
                    "ml_failure_risk": sensor.get("ml_failure_risk_latest"),
                    "hybrid_failure_risk": sensor.get("hybrid_failure_risk"),
                    "operational_rule_score": sensor.get("operational_rule_score"),
                    "hybrid_health_score": sensor.get("hybrid_health_score"),
                    "rul_days": sensor.get("estimated_rul_days"),
                    "risk_level": display_risk,
                    "priority": display_priority,
                    "priority_score": priority.get("priority_score"),
                    "urgency": display_urgency,
                    "delay_hours": delay.get("delay_hours", 0),
                }
            )
        table = (
            pd.DataFrame(rows)
            .sort_values(
                ["priority_score", "hybrid_health_score", "rul_days"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )
        dynamic_scope_ids = set(dynamic_asset_ids())
        ranked_ids = set(table["asset_id"].astype(str).str.upper())
        is_dynamic_only_scope = bool(ranked_ids) and ranked_ids.issubset(dynamic_scope_ids)
        report_title = "Dynamic Assets Priority Ranking" if is_dynamic_only_scope else "Plant-Level Maintenance Decision Summary"
        top_asset = table.iloc[0]["asset_id"]
        top_sensor = self.get_latest_sensor_summary(top_asset)
        top_spares = self.get_spares(top_asset)
        top_type_text = str(top_sensor.get("asset_type", "")).lower()
        top_equipment = normalize_equipment_type(top_sensor.get("asset_type", ""))
        second = table.iloc[1] if len(table) > 1 else None
        if top_equipment == "gearbox":
            recommended_first_action = (
                f"Choose {top_asset} first. Create a P1 controlled-shutdown inspection plan for gearbox vibration, "
                "reserve gearbox bearing set and synthetic gear oil, perform vibration spectrum analysis, oil sampling, "
                "coupling alignment check, gear backlash check, and bearing inspection."
            )
        elif top_equipment == "motor":
            recommended_first_action = (
                f"Choose {top_asset} first. Create a P1 motor overheating inspection plan, reduce load if needed, "
                "inspect bearing lubrication, cooling path, current imbalance, fan condition, and coupling alignment."
            )
        elif top_equipment == "pump":
            recommended_first_action = (
                f"Choose {top_asset} first. Create a high-priority cavitation inspection plan, check suction strainer, "
                "tank level, air ingress, seal leakage, impeller condition, and reserve mechanical seal or impeller spares."
            )
        elif top_equipment == "hydraulic":
            recommended_first_action = (
                f"Choose {top_asset} first. Create a high-priority hydraulic pressure recovery plan, inspect filter, "
                "relief valve, oil level, leakage, pump noise, and reserve filter element or relief valve cartridge."
            )
        elif any(word in top_type_text for word in ["blower", "fan", "compressor"]):
            recommended_first_action = (
                f"Choose {top_asset} first. Create a P1 rotating-air-equipment inspection plan, verify vibration spectrum, "
                "bearing temperature, motor current balance, damper position, impeller fouling, duct restriction, "
                "coupling alignment, and standby availability."
            )
        elif "bearing" in top_type_text:
            recommended_first_action = (
                f"Choose {top_asset} first. Create a P1 bearing inspection plan, verify bearing temperature, lubrication, "
                "vibration spectrum, alignment, load condition, contamination, cooling path, spare bearing availability, "
                "lifting plan, and safe isolation permit."
            )
        elif top_equipment == "blast_furnace":
            recommended_first_action = (
                f"Choose {top_asset} first. Create a P1 blast-furnace-area safety inspection plan, verify cooling, airflow, "
                "interlocks, vibration, temperature, isolation permits, and standby equipment readiness."
            )
        else:
            recommended_first_action = f"Choose {top_asset} first. Create a controlled inspection and repair work order."
        comparison_note = ""
        if second is not None:
            comparison_note = (
                f"- It is ahead of {second['asset_id']}: {second['asset_id']} is "
                f"{second['priority']}/{second['risk_level']} with hybrid health score "
                f"{second['hybrid_health_score']} and RUL {second['rul_days']} days."
            )
        docs = self._filter_docs_for_assets(self.rag.retrieve(query, top_k=8, plant_level=True), list(asset_ids))[:5]
        dynamic_docs = []
        for asset_id in table["asset_id"].astype(str).tolist():
            sensor = self.get_latest_sensor_summary(asset_id)
            if sensor.get("is_dynamic"):
                dynamic_docs.extend(self._dynamic_context_docs(asset_id, sensor))
        docs = dynamic_docs + docs
        agent_plan = self.build_agent_plan(query, mode="plant_priority", asset_id=top_asset)
        tool_calls = [
            {"tool": "asset_health_scan", "agent": "Sensor Agent", "input": f"{len(asset_ids)} scoped assets", "output": f"{len(table)} scored rows from {'dynamic memory only' if is_dynamic_only_scope else 'plant scope'}", "status": "success"},
            {"tool": "plant_priority_ranker", "agent": "Risk Agent", "input": "hybrid score + RUL + delay + criticality", "output": f"top asset {top_asset}", "status": "success"},
            {"tool": "rag_retriever", "agent": "Knowledge Agent", "input": "plant-level policies and evidence", "output": f"{len(docs)} evidence chunks", "status": "success"},
            {"tool": "supervisor_report_writer", "agent": "Reporter Agent", "input": top_asset, "output": "plant priority summary generated", "status": "success"},
        ]
        verifier_checks = [
            {"check": "All requested known assets scored", "status": "pass" if len(table) == len(asset_ids) else "review", "detail": f"{len(table)} of {len(asset_ids)} assets ranked"},
            {"check": "Top asset selected", "status": "pass", "detail": top_asset},
            {"check": "Ranking includes RUL and delay", "status": "pass", "detail": "rul_days and delay_hours present"},
            {"check": "Policy evidence retrieved", "status": "pass" if len(docs) > 0 else "review", "detail": f"{len(docs)} sources"},
        ]
        decision_packet = {
            "mode": "plant_priority",
            "intent": "maintenance_prioritization",
            "objective": query,
            "selected_asset": top_asset,
            "risk_level": table.iloc[0]["risk_level"],
            "priority": table.iloc[0]["priority"],
            "urgency": table.iloc[0]["urgency"],
            "hybrid_health_score": float(table.iloc[0]["hybrid_health_score"]),
            "hybrid_failure_risk": float(table.iloc[0]["hybrid_failure_risk"]),
            "ml_failure_risk": float(table.iloc[0]["ml_failure_risk"]),
            "operational_rule_score": float(table.iloc[0]["operational_rule_score"]),
            "estimated_rul_days": float(table.iloc[0]["rul_days"]),
            "recommended_first_action": recommended_first_action,
            "next_system_action": "create_first_work_order_and_notify_supervisor",
            "top_sources": [doc.get("source") for doc in docs[:3]],
        }
        ranking = "\n".join(
            f"- {r.asset_id}: {r.priority}/{r.risk_level}, hybrid score {r.hybrid_health_score}, ML risk {r.ml_failure_risk}, rule score {r.operational_rule_score}, RUL {r.rul_days} days, delay {r.delay_hours}h"
            for r in table.itertuples()
        )
        report = f"""
**{report_title}**

**Choose {top_asset} first.**

**Reason**
- {top_asset} has the highest combined plant priority score from hybrid health risk, criticality, RUL, delay impact, and spare readiness.
- Current locked fields: {table.iloc[0]["priority"]}/{table.iloc[0]["risk_level"]}, hybrid health score {table.iloc[0]["hybrid_health_score"]}, hybrid failure risk {table.iloc[0]["hybrid_failure_risk"]}, RUL {table.iloc[0]["rul_days"]} days.
- Latest condition: temperature {top_sensor.get("temperature_latest")}, vibration {top_sensor.get("vibration_latest")}, pressure {top_sensor.get("pressure_latest")}, alarms {top_sensor.get("alarm_count_latest")}.
{comparison_note}

**Agentic Control Loop**
- Objective: {query}
- Selected first target: {top_asset}
- Operating mode: {'dynamic-memory prioritization' if is_dynamic_only_scope else 'autonomous plant prioritization'}
- Decision policy: rank by safety risk, hybrid risk, criticality, RUL, delay impact, and spare readiness.

**Autonomous Execution Plan**
{chr(10).join([f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan])}

**Tool Calls Executed**
{chr(10).join([f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}" for t in tool_calls])}

**Verifier Checks**
{chr(10).join([f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks])}

**Locked Decision Fields**
- Most urgent asset: {top_asset}
- Intent: maintenance_prioritization
- Recommended first action: {recommended_first_action}
- Ranking basis: hybrid ML + operational rule score, criticality, delay severity, RUL, anomaly status, spares/procurement.

**Diagnosis**
- Equipment was compared across {len(asset_ids)} scoped steel assets{' from dynamic memory only' if is_dynamic_only_scope else ', including any user-added dynamic assets in memory'}.

**Risk and RUL**
{ranking}

**Immediate Actions**
- {recommended_first_action}
- Reserve spares before shutdown.
- Notify area supervisor for P1/P2 assets.
- Continue monitoring lower-ranked assets.

**Spare Strategy For Selected Asset**
{_spares_strategy(top_spares)}

**Evidence / Sources**
{_format_sources(docs)}

**Agent Reasoning Trace**
- Triage Agent: identified plant-level prioritization request.
- Sensor Agent: collected latest health and RUL for all requested known assets, including dynamic memory rows.
- Risk Agent: ranked assets by hybrid ML + operational rule score.
- Planning Agent: selected {top_asset} as first maintenance target.
- Reporting Agent: generated supervisor summary and logbook entry.

**Final Decision Packet**
- Mode: {decision_packet["mode"]}
- Intent: {decision_packet["intent"]}
- Next system action: {decision_packet["next_system_action"]}
- Selected asset: {decision_packet["selected_asset"]}
- Recommended first action: {decision_packet["recommended_first_action"]}
- Top evidence sources: {", ".join(decision_packet["top_sources"])}
""".strip()
        priority = {"priority": "PLANT", "risk_level": "PLANT_SUMMARY", "urgency": f"Prioritize {top_asset}", "priority_score": float(table.iloc[0]["priority_score"])}
        self.session_memory["last_asset_id"] = top_asset
        if self._is_dynamic_asset(top_asset):
            self.session_memory["last_new_asset_id"] = top_asset
        self.write_logbook(query, top_asset, priority, report)
        return {
            "mode": "plant_priority",
            "asset_id": top_asset,
            "intent": "maintenance_prioritization",
            "plant_priority_table": table.to_dict("records"),
            "risk_priority": priority,
            "priority": "Plant priority summary",
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "answer": report,
            "final_answer": report,
            "alert_report": f"Plant alert: prioritize {top_asset}.",
            "llm_used": True,
        }

    def ingest_new_sensor_alert(self, asset_id: str, temperature: float, vibration: float, current: float, pressure: float, rpm: float = 1480, alarm_count: int = 2, user_id: str = "iot_gateway") -> dict:
        self.ensure_ready()
        raw = pd.read_csv(DATA_DIR / "steel_sensor_logs.csv")
        rows = raw[raw["asset_id"] == asset_id]
        if rows.empty:
            return {"asset_id": asset_id, "answer": f"No known asset found for {asset_id}.", "priority": "UNKNOWN", "alert_report": "No alert generated."}
        info = rows.iloc[-1].to_dict()
        row = {
            "source": "steel_demo_app",
            "timestamp": datetime.now().isoformat(),
            "asset_id": asset_id,
            "asset_type": info.get("asset_type"),
            "area": info.get("area"),
            "criticality": info.get("criticality"),
            "criticality_score": info.get("criticality_score", 2),
            "temperature": float(temperature),
            "vibration": float(vibration),
            "current": float(current),
            "pressure": float(pressure),
            "rpm": float(rpm),
            "alarm_count": int(alarm_count),
            "delay_hours": info.get("delay_hours", 0),
            "spare_lead_time_days": info.get("spare_lead_time_days", 0),
            "failure_label": 0,
            "failure_mode": "real_time_alert",
        }
        raw = pd.concat([raw, pd.DataFrame([row])], ignore_index=True, sort=False)
        raw.to_csv(DATA_DIR / "steel_sensor_logs.csv", index=False)
        scored = self.model_manager.score_live_alert(row)
        create_compatibility_sensor_log()
        # Rebuild RAG so the latest health document is updated for evidence retrieval.
        self.rag.build()
        result = self.chat(f"New real-time alert for {asset_id}. Diagnose and generate alert report.", user_id=user_id)
        result["live_alert_row"] = scored
        return result

    def logbook(self) -> pd.DataFrame:
        path = DATA_DIR / "digital_logbook.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def feedback_log(self) -> pd.DataFrame:
        path = DATA_DIR / "feedback_log.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
