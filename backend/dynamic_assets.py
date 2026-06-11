"""Dynamic asset memory and scoring for user-added plant equipment."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DATA_DIR


DYNAMIC_ASSET_PATH = DATA_DIR / "dynamic_assets.csv"

DYNAMIC_ASSET_COLUMNS = [
    "asset_id",
    "asset_type",
    "area",
    "criticality",
    "temperature",
    "vibration",
    "current",
    "pressure",
    "rpm",
    "alarm_count",
    "delay_hours",
    "spare_lead_time_days",
    "created_at",
    "updated_at",
    "source_query",
]


def _clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _extract_text(patterns: list[str], text: str, default: str = "") -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip(" :,-.;")
            if value:
                return value
    return default


def _extract_number(patterns: list[str], text: str, default: float | None = None) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return default


def load_dynamic_assets() -> pd.DataFrame:
    if DYNAMIC_ASSET_PATH.exists():
        df = pd.read_csv(DYNAMIC_ASSET_PATH)
        for col in DYNAMIC_ASSET_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        return df[DYNAMIC_ASSET_COLUMNS].copy()
    return pd.DataFrame(columns=DYNAMIC_ASSET_COLUMNS)


def save_dynamic_assets(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for col in DYNAMIC_ASSET_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df[DYNAMIC_ASSET_COLUMNS].to_csv(DYNAMIC_ASSET_PATH, index=False)


def dynamic_asset_ids() -> list[str]:
    df = load_dynamic_assets()
    if df.empty:
        return []
    return sorted(df["asset_id"].dropna().astype(str).str.upper().unique().tolist())


def is_asset_ingestion_query(query: str) -> bool:
    q = str(query).lower()
    has_asset_id = bool(re.search(r"\b[A-Z]{2,6}-\d{1,5}\b", str(query), flags=re.IGNORECASE))
    add_terms = [
        "add a new asset",
        "add new asset",
        "add asset",
        "register asset",
        "register a new asset",
        "remember this asset",
        "remember these assets",
        "remember all",
        "ingest asset",
        "create asset",
        "new asset:",
    ]
    return has_asset_id and any(term in q for term in add_terms)


def query_mentions_new_asset_reference(query: str) -> bool:
    q = str(query).lower()
    return any(
        term in q
        for term in [
            "same new asset",
            "that new asset",
            "the new asset",
            "newly added asset",
            "newly added assets",
            "added asset",
            "added assets",
        ]
    )


def extract_asset_ids(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for match in re.finditer(r"\b[A-Z]{2,6}-\d{1,5}\b", str(text), flags=re.IGNORECASE):
        asset_id = match.group(0).upper()
        if asset_id not in seen:
            seen.add(asset_id)
            out.append(asset_id)
    return out


def _segment_by_asset_id(query: str) -> list[tuple[str, str]]:
    ids = list(re.finditer(r"\b[A-Z]{2,6}-\d{1,5}\b", query, flags=re.IGNORECASE))
    segments: list[tuple[str, str]] = []
    for idx, match in enumerate(ids):
        start = match.start()
        end = ids[idx + 1].start() if idx + 1 < len(ids) else len(query)
        segment = query[start:end]
        segments.append((match.group(0).upper(), segment))
    return segments


def parse_dynamic_assets(query: str) -> list[dict]:
    """Extract one or more user-specified assets from natural language."""

    assets: list[dict] = []
    now = datetime.now().isoformat(timespec="seconds")
    for asset_id, segment in _segment_by_asset_id(str(query)):
        asset_type = _extract_text(
            [
                r"(?:asset\s*type|equipment\s*type|type)\s*(?:is|=|:)?\s*([^,;\n.]+)",
                r"(?:asset|equipment)\s+" + re.escape(asset_id) + r"\s*(?:is|:)?\s*([^,;\n.]+)",
            ],
            segment,
            "Unspecified Equipment",
        )
        area = _extract_text([r"\barea\s*(?:is|=|:)?\s*([^,;\n.]+)"], segment, "Unspecified Area")
        criticality = _extract_text(
            [r"\bcriticality\s*(?:is|=|:)?\s*(critical|high|medium|low)", r"\b(critical|high|medium|low)\s+criticality\b"],
            segment,
            "Medium",
        ).title()

        asset = {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "area": area,
            "criticality": criticality,
            "temperature": _extract_number([r"\btemp(?:erature)?\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)"], segment, np.nan),
            "vibration": _extract_number([r"\bvib(?:ration)?\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)"], segment, np.nan),
            "current": _extract_number([r"\bcurrent\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)"], segment, np.nan),
            "pressure": _extract_number([r"\bpressure\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)"], segment, np.nan),
            "rpm": _extract_number([r"\brpm\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)"], segment, 1480.0),
            "alarm_count": _extract_number([r"\balarm\s*count\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)", r"\balarms?\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)"], segment, 0.0),
            "delay_hours": _extract_number([r"\bdelay(?:\s*hours)?\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)"], segment, 0.0),
            "spare_lead_time_days": _extract_number([r"\blead\s*time(?:\s*days)?\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)"], segment, 0.0),
            "created_at": now,
            "updated_at": now,
            "source_query": str(query),
        }
        assets.append(asset)
    return assets


def upsert_dynamic_assets(assets: list[dict]) -> pd.DataFrame:
    current = load_dynamic_assets()
    if not assets:
        return current

    incoming = pd.DataFrame(assets)
    for col in DYNAMIC_ASSET_COLUMNS:
        if col not in incoming.columns:
            incoming[col] = np.nan
    incoming["asset_id"] = incoming["asset_id"].astype(str).str.upper()

    keep = current[~current["asset_id"].astype(str).str.upper().isin(set(incoming["asset_id"]))].copy()
    if keep.empty:
        merged = incoming[DYNAMIC_ASSET_COLUMNS].copy()
    else:
        merged = pd.concat([keep, incoming[DYNAMIC_ASSET_COLUMNS]], ignore_index=True, sort=False)
    save_dynamic_assets(merged)
    return merged


def criticality_score(criticality: Any) -> int:
    c = str(criticality).strip().lower()
    if c == "critical":
        return 3
    if c == "high":
        return 2
    if c == "medium":
        return 1
    return 0


def score_dynamic_asset(row: pd.Series | dict) -> dict:
    r = dict(row)
    asset_type = _clean_text(r.get("asset_type"), "Unspecified Equipment")
    typ = asset_type.lower()
    criticality = _clean_text(r.get("criticality"), "Medium").title()
    temp = float(pd.to_numeric(r.get("temperature", np.nan), errors="coerce") if pd.notna(r.get("temperature", np.nan)) else 0)
    vib = float(pd.to_numeric(r.get("vibration", np.nan), errors="coerce") if pd.notna(r.get("vibration", np.nan)) else 0)
    current = float(pd.to_numeric(r.get("current", np.nan), errors="coerce") if pd.notna(r.get("current", np.nan)) else 0)
    pressure = float(pd.to_numeric(r.get("pressure", np.nan), errors="coerce") if pd.notna(r.get("pressure", np.nan)) else 8)
    alarms = float(pd.to_numeric(r.get("alarm_count", np.nan), errors="coerce") if pd.notna(r.get("alarm_count", np.nan)) else 0)

    score = 0.0
    if temp >= 70:
        score += 8
    if temp >= 80:
        score += 20
    if temp >= 90:
        score += 15
    if vib >= 5:
        score += 10
    if vib >= 6:
        score += 20
    if vib >= 8:
        score += 20
    if current >= 75:
        score += 8
    if current >= 85:
        score += 15
    if pressure <= 6:
        score += 18
    if alarms >= 2:
        score += 10
    if alarms >= 3:
        score += 10

    crit = criticality.lower()
    if crit == "critical":
        score += 15
    elif crit == "high":
        score += 10
    elif crit == "medium":
        score += 5

    if "blast furnace" in (typ + " " + str(r.get("area", "")).lower()) and vib >= 6.5 and temp >= 80:
        score += 20
    if any(word in typ for word in ["blower", "fan", "compressor"]) and vib >= 6 and current >= 80:
        score += 12
    if "hydraulic" in typ and pressure <= 6:
        score += 10
    if "pump" in typ and pressure <= 6:
        score += 10

    score = float(np.clip(score, 0, 100))
    if score >= 75:
        risk = "CRITICAL"
        priority = "P1"
        urgency = "Immediate action"
    elif score >= 55:
        risk = "HIGH"
        priority = "P2"
        urgency = "Action within 24 hours"
    elif score >= 35:
        risk = "MEDIUM"
        priority = "P3"
        urgency = "Plan in maintenance window"
    else:
        risk = "LOW"
        priority = "P4"
        urgency = "Monitor only"

    rul = max(1.0, round(50 * (1 - score / 100), 1))
    ml_risk_proxy = round(float(np.clip(0.20 + 0.0065 * score, 0.01, 0.92)), 4)

    return {
        **r,
        "asset_id": str(r.get("asset_id", "")).upper(),
        "asset_type": asset_type,
        "area": _clean_text(r.get("area"), "Unspecified Area"),
        "criticality": criticality,
        "criticality_score": criticality_score(criticality),
        "temperature": temp,
        "vibration": vib,
        "current": current,
        "pressure": pressure,
        "rpm": float(pd.to_numeric(r.get("rpm", 1480), errors="coerce") or 1480),
        "alarm_count": int(float(pd.to_numeric(r.get("alarm_count", 0), errors="coerce") or 0)),
        "anomaly_score": round(score / 100, 4),
        "is_anomaly": int(score >= 55),
        "failure_risk": round(score / 100, 4),
        "ml_failure_risk": ml_risk_proxy,
        "ml_failure_pred": int(score >= 55),
        "operational_rule_score": round(score, 2),
        "hybrid_health_score": round(score, 2),
        "hybrid_failure_risk": round(score / 100, 4),
        "risk_band": risk,
        "priority": priority,
        "urgency": urgency,
        "estimated_rul_days": rul,
        "anomaly_events_24h": int(max(alarms * 4, 1 if score >= 55 else 0)),
        "temperature_slope_24h": 0.0,
        "vibration_slope_24h": 0.0,
        "pressure_slope_24h": 0.0,
        "data_origin": "dynamic_user_memory",
        "is_dynamic": 1,
    }


def score_dynamic_assets(df: pd.DataFrame | None = None) -> pd.DataFrame:
    source = load_dynamic_assets() if df is None else df.copy()
    if source.empty:
        return pd.DataFrame()
    return pd.DataFrame([score_dynamic_asset(row) for _, row in source.iterrows()])


def dynamic_root_cause(asset_type: str) -> str:
    typ = str(asset_type).lower()
    if "blower" in typ or "fan" in typ:
        return "bearing wear, impeller imbalance, fouling, duct restriction, motor overload, or coupling misalignment"
    if "bearing" in typ:
        return "bearing wear, lubrication breakdown, misalignment, overload, contamination, or inadequate cooling"
    if "pump" in typ:
        return "low suction head, clogged strainer, seal wear, bearing wear, impeller erosion, or air ingress"
    if "compressor" in typ:
        return "bearing wear, surge condition, filter restriction, lubrication issue, or coupling misalignment"
    if "conveyor" in typ:
        return "belt mistracking, pulley bearing wear, material buildup, gearbox issue, or idler seizure"
    if "furnace" in typ:
        return "cooling restriction, airflow instability, refractory hot spot, bearing overload, or process imbalance"
    return "abnormal operating condition inferred from current readings, alarm state, criticality, and equipment class"


def dynamic_actions(asset_type: str) -> list[str]:
    typ = str(asset_type).lower()
    if "blower" in typ or "fan" in typ:
        return [
            "Verify blower vibration spectrum, bearing temperature, motor current balance, inlet/outlet damper position, and foundation bolts.",
            "Check impeller fouling, duct restriction, lubrication condition, coupling alignment, and standby blower availability.",
        ]
    if "bearing" in typ:
        return [
            "Check bearing temperature, lubrication condition, vibration spectrum, alignment, loading, contamination, and cooling path.",
            "Prepare controlled inspection with spare bearing set, seals, lifting plan, and restart acceptance readings.",
        ]
    if "pump" in typ:
        return [
            "Inspect suction strainer, suction head, inlet valve position, seal leakage, bearing condition, and impeller erosion.",
        ]
    if "compressor" in typ:
        return [
            "Check suction filter, discharge pressure trend, bearing temperature, lubrication, surge alarms, and coupling alignment.",
        ]
    if "conveyor" in typ:
        return [
            "Inspect belt tracking, pulley bearings, idlers, take-up tension, gearbox oil, and emergency pull-cord status.",
        ]
    if "furnace" in typ:
        return [
            "Check cooling flow, hot-spot trend, airflow stability, refractory condition, interlock alarms, and safe isolation plan.",
        ]
    return ["Inspect the asset boundary, confirm sensor readings, isolate energy sources if required, and create a maintenance work order."]


def dynamic_spares(asset_id: str, asset_type: str) -> list[dict]:
    typ = str(asset_type).lower()
    if "blower" in typ or "fan" in typ:
        parts = [
            ("Blower bearing set", 1, 10, "Critical", 95000),
            ("Coupling insert/alignment kit", 2, 3, "High", 18000),
        ]
    elif "compressor" in typ:
        parts = [
            ("Compressor bearing kit", 1, 14, "Critical", 120000),
            ("Suction filter element", 3, 4, "High", 12000),
        ]
    elif "conveyor" in typ:
        parts = [
            ("Pulley bearing set", 2, 7, "High", 32000),
            ("Idler roller set", 8, 2, "Medium", 5000),
        ]
    else:
        parts = [
            ("Inspection consumables and isolation kit", 1, 1, "Medium", 5000),
            ("Equipment-specific critical spare to verify", 0, 7, "Critical", 0),
        ]
    return [
        {
            "asset_id": asset_id,
            "spare_part": spare,
            "stock_qty": qty,
            "lead_time_days": lead,
            "spare_criticality": crit,
            "unit_cost_inr": cost,
            "source": "dynamic_spares_inference",
        }
        for spare, qty, lead, crit, cost in parts
    ]
