"""Dynamic asset memory, updates, and scoring for user-added plant equipment."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import DATA_DIR


DYNAMIC_ASSET_PATH = DATA_DIR / "dynamic_assets.csv"
DYNAMIC_ASSET_HISTORY_PATH = DATA_DIR / "dynamic_asset_history.csv"

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
    "operator_notes",
    "created_at",
    "updated_at",
    "source_query",
]

DYNAMIC_HISTORY_COLUMNS = [
    "event_id",
    "event_type",
    "asset_id",
    "changed_at",
    "changed_fields",
    "previous_score",
    "new_score",
    "previous_priority",
    "new_priority",
    "previous_risk_band",
    "new_risk_band",
    "source_query",
    "previous_record",
    "new_record",
]

NUMERIC_FIELDS = [
    "temperature",
    "vibration",
    "current",
    "pressure",
    "rpm",
    "alarm_count",
    "delay_hours",
    "spare_lead_time_days",
]

READING_CONNECTOR = (
    r"(?:\s+(?:is|was|has|have|now|changed|increased|decreased|reduced|improved|"
    r"dropped|rose|fell|went|became|set|updated))*\s*(?:=|:|to)?\s*"
)


def _clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    text = str(value).strip()
    return text if text else default


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _num(value: Any, default: float = np.nan) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return default
    return float(parsed)


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


def _reading_pattern(label: str) -> str:
    return rf"\b{label}{READING_CONNECTOR}(-?\d+(?:\.\d+)?)"


def _extract_operator_notes(text: str) -> str:
    notes: list[str] = []
    patterns = [
        r"\b(?:operator|technician|engineer|shift team|maintenance team)\s+(?:reports?|reported|observes?|observed|mentions?|mentioned|hears?|heard|sees?|saw)\s+([^.;\n]+)",
        r"\b(?:symptom|observation|operator note|field note|note)\s*(?:is|=|:)?\s*([^.;\n]+)",
        r"\b(?:but|and)\s+(?:there is|there are|there was|there were|it has|asset has|shows?|showing)\s+([^.;\n]*(?:noise|cavitation|leak|smoke|smell|surge|sparking|overheat|overheating|vibration|chatter|rubbing)[^.;\n]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            note = re.sub(r"\s+", " ", match.group(1)).strip(" ,:-")
            if note and note.lower() not in {n.lower() for n in notes}:
                notes.append(note)
    lowered = text.lower()
    symptom_phrases = [
        "loud cavitation noise",
        "cavitation noise",
        "abnormal noise",
        "bearing noise",
        "burning smell",
        "oil leak",
        "hydraulic leak",
        "smoke",
        "sparking",
        "surging",
    ]
    for phrase in symptom_phrases:
        existing = {n.lower() for n in notes}
        if phrase in lowered and phrase not in existing and not any(phrase in note or note in phrase for note in existing):
            notes.append(phrase)
    return "; ".join(notes)


def _json_clean(record: dict) -> dict:
    clean: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (np.integer,)):
            clean[key] = int(value)
        elif isinstance(value, (np.floating,)):
            clean[key] = None if np.isnan(value) else float(value)
        elif isinstance(value, float) and np.isnan(value):
            clean[key] = None
        elif pd.isna(value) if not isinstance(value, (list, dict, tuple, set)) else False:
            clean[key] = None
        else:
            clean[key] = value
    return clean


def load_dynamic_assets() -> pd.DataFrame:
    if DYNAMIC_ASSET_PATH.exists():
        df = pd.read_csv(DYNAMIC_ASSET_PATH)
        for col in DYNAMIC_ASSET_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        for col in NUMERIC_FIELDS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["asset_id", "asset_type", "area", "criticality", "operator_notes", "created_at", "updated_at", "source_query"]:
            df[col] = df[col].astype("object")
        df["asset_id"] = df["asset_id"].astype(str).str.upper()
        return df[DYNAMIC_ASSET_COLUMNS].copy()
    return pd.DataFrame(columns=DYNAMIC_ASSET_COLUMNS)


def save_dynamic_assets(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for col in DYNAMIC_ASSET_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    out = df[DYNAMIC_ASSET_COLUMNS].copy()
    for col in ["asset_id", "asset_type", "area", "criticality", "operator_notes", "created_at", "updated_at", "source_query"]:
        out[col] = out[col].astype("object")
    out["asset_id"] = out["asset_id"].astype(str).str.upper()
    out.to_csv(DYNAMIC_ASSET_PATH, index=False)


def load_dynamic_asset_history() -> pd.DataFrame:
    if DYNAMIC_ASSET_HISTORY_PATH.exists():
        df = pd.read_csv(DYNAMIC_ASSET_HISTORY_PATH)
        for col in DYNAMIC_HISTORY_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[DYNAMIC_HISTORY_COLUMNS].copy()
    return pd.DataFrame(columns=DYNAMIC_HISTORY_COLUMNS)


def append_dynamic_asset_history(rows: list[dict]) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    current = load_dynamic_asset_history()
    if not rows:
        return current
    incoming = pd.DataFrame(rows)
    for col in DYNAMIC_HISTORY_COLUMNS:
        if col not in incoming.columns:
            incoming[col] = ""
    merged = pd.concat([current, incoming[DYNAMIC_HISTORY_COLUMNS]], ignore_index=True, sort=False)
    merged.to_csv(DYNAMIC_ASSET_HISTORY_PATH, index=False)
    return merged


def latest_dynamic_asset_change(asset_id: str) -> dict | None:
    history = load_dynamic_asset_history()
    if history.empty:
        return None
    rows = history[
        (history["asset_id"].astype(str).str.upper() == str(asset_id).upper())
        & (history["event_type"].astype(str).str.lower() == "update")
    ].copy()
    if rows.empty:
        return None
    rows["changed_at_sort"] = pd.to_datetime(rows["changed_at"], errors="coerce", format="mixed")
    row = rows.sort_values("changed_at_sort").iloc[-1].to_dict()
    for key in ["previous_record", "new_record"]:
        try:
            row[key] = json.loads(row.get(key) or "{}")
        except json.JSONDecodeError:
            row[key] = {}
    try:
        row["changed_fields"] = json.loads(row.get("changed_fields") or "[]")
    except json.JSONDecodeError:
        row["changed_fields"] = []
    return row


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
        "register new asset",
        "register new assets",
        "register three new assets",
        "remember this asset",
        "remember these assets",
        "remember all",
        "ingest asset",
        "create asset",
        "new asset:",
    ]
    return has_asset_id and any(term in q for term in add_terms)


def is_asset_update_query(query: str) -> bool:
    if is_asset_ingestion_query(query):
        return False
    q = str(query).lower()
    has_asset_id = bool(extract_asset_ids(query))
    update_terms = [
        "update",
        "change",
        " set ",
        "new reading",
        "new readings",
        "latest reading",
        "latest readings",
        "reading changed",
        "readings changed",
        "increased to",
        "decreased to",
        "now ",
    ]
    reading_terms = ["temperature", "temp", "vibration", "vib", "current", "pressure", "alarm", "rpm"]
    return has_asset_id and any(term in q for term in update_terms) and any(term in q for term in reading_terms)


def is_priority_change_query(query: str) -> bool:
    q = str(query).lower()
    return any(
        term in q
        for term in [
            "priority change",
            "priority changed",
            "did priority change",
            "what changed after",
            "after the update",
            "after update",
            "changed after update",
        ]
    )


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


def _infer_asset_type(asset_id: str, segment: str, explicit: str) -> str:
    if explicit and explicit.lower() not in {"unspecified equipment", "unknown", "not provided"}:
        return explicit
    tail = re.sub(r"^\s*(?:id\s*)?" + re.escape(asset_id) + r"\b", "", segment, flags=re.IGNORECASE).strip(" ,:-")
    tail = re.split(
        r"\b(?:area|criticality|temperature|temp|vibration|vib|current|pressure|rpm|alarm|alarms|available readings|readings|spares|spare)\b",
        tail,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    candidate = re.sub(r"\s+", " ", tail).strip(" ,:-.;")
    if len(candidate) >= 3 and not re.search(r"\d", candidate):
        return candidate.title()

    prefix = asset_id.split("-", 1)[0]
    prefix_map = {
        "BF": "Blast Furnace Equipment",
        "GBX": "Gearbox",
        "MTR": "Motor",
        "PMP": "Pump",
        "HPP": "Hydraulic Power Pack",
        "RMF": "Roughing Mill Fan",
        "DES": "Descaler Pump",
        "CCM": "Caster Mold Oscillator",
        "SNT": "Sinter Plant Exhaust Fan",
    }
    return prefix_map.get(prefix, "Unspecified Equipment")


def _infer_area(asset_id: str, asset_type: str, explicit: str) -> str:
    if explicit and explicit.lower() not in {"unspecified area", "unknown", "not provided"}:
        return explicit.title()
    text = f"{asset_id} {asset_type}".lower()
    if "sinter" in text:
        return "Sinter Plant"
    if "roughing mill" in text or asset_id.startswith("RMF-"):
        return "Roughing Mill"
    if "descaler" in text or asset_id.startswith("DES-"):
        return "Descaling"
    if "caster" in text or asset_id.startswith("CCM-"):
        return "Caster"
    if "blast furnace" in text or asset_id.startswith("BF-"):
        return "Blast Furnace"
    if "finishing mill" in text or "gearbox" in text:
        return "Finishing Mill"
    if "hydraulic" in text:
        return "Hydraulic Station"
    if "pump" in text:
        return "Utility"
    return "Unspecified Area"


def _criticality_from_segment(segment: str) -> str:
    explicit = _extract_text(
        [
            r"\bcriticality\s*(?:is|=|:)?\s*(critical|high|medium|low)",
            r"\b(critical|high|medium|low)\s+criticality\b",
        ],
        segment,
        "",
    )
    if explicit:
        return explicit.title()
    return "Medium"


def parse_dynamic_assets(query: str) -> list[dict]:
    """Extract one or more user-specified assets from natural language."""

    assets: list[dict] = []
    now = datetime.now().isoformat(timespec="seconds")
    for asset_id, segment in _segment_by_asset_id(str(query)):
        explicit_type = _extract_text(
            [
                r"(?:asset\s*type|equipment\s*type|type)\s*(?:is|=|:)?\s*([^,;\n.]+)",
                r"(?:asset|equipment)\s+" + re.escape(asset_id) + r"\s*(?:is|:)?\s*([^,;\n.]+)",
            ],
            segment,
            "",
        )
        asset_type = _infer_asset_type(asset_id, segment, explicit_type)
        explicit_area = _extract_text([r"\barea\s*(?:is|=|:)?\s*([^,;\n.]+)"], segment, "")
        area = _infer_area(asset_id, asset_type, explicit_area)

        asset = {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "area": area,
            "criticality": _criticality_from_segment(segment),
            "temperature": _extract_number([_reading_pattern(r"temp(?:erature)?")], segment, np.nan),
            "vibration": _extract_number([_reading_pattern(r"vib(?:ration)?")], segment, np.nan),
            "current": _extract_number([_reading_pattern(r"current")], segment, np.nan),
            "pressure": _extract_number([_reading_pattern(r"pressure")], segment, np.nan),
            "rpm": _extract_number([_reading_pattern(r"rpm")], segment, 1480.0),
            "alarm_count": _extract_number(
                [
                    _reading_pattern(r"alarm\s*count"),
                    _reading_pattern(r"alarms?"),
                ],
                segment,
                0.0,
            ),
            "delay_hours": _extract_number([_reading_pattern(r"delay(?:\s*hours)?")], segment, 0.0),
            "spare_lead_time_days": _extract_number([_reading_pattern(r"lead\s*time(?:\s*days)?")], segment, 0.0),
            "operator_notes": _extract_operator_notes(segment),
            "created_at": now,
            "updated_at": now,
            "source_query": str(query),
        }
        assets.append(asset)
    return assets


def parse_dynamic_asset_updates(query: str) -> list[dict]:
    updates: list[dict] = []
    now = datetime.now().isoformat(timespec="seconds")
    for asset_id, segment in _segment_by_asset_id(str(query)):
        fields: dict[str, Any] = {}
        patterns = {
            "temperature": [_reading_pattern(r"temp(?:erature)?")],
            "vibration": [_reading_pattern(r"vib(?:ration)?")],
            "current": [_reading_pattern(r"current")],
            "pressure": [_reading_pattern(r"pressure")],
            "rpm": [_reading_pattern(r"rpm")],
            "alarm_count": [
                _reading_pattern(r"alarm\s*count"),
                _reading_pattern(r"alarms?"),
            ],
        }
        for field, pats in patterns.items():
            value = _extract_number(pats, segment, None)
            if value is not None:
                fields[field] = value
        note = _extract_operator_notes(segment)
        if note:
            fields["operator_notes"] = note
        if fields:
            updates.append({"asset_id": asset_id, "fields": fields, "updated_at": now, "source_query": str(query)})
    return updates


def upsert_dynamic_assets(assets: list[dict]) -> pd.DataFrame:
    current = load_dynamic_assets()
    if not assets:
        return current

    incoming = pd.DataFrame(assets)
    for col in DYNAMIC_ASSET_COLUMNS:
        if col not in incoming.columns:
            incoming[col] = np.nan
    incoming["asset_id"] = incoming["asset_id"].astype(str).str.upper()

    rows: list[dict] = []
    for _, new_row in incoming[DYNAMIC_ASSET_COLUMNS].iterrows():
        asset_id = str(new_row["asset_id"]).upper()
        previous = current[current["asset_id"].astype(str).str.upper() == asset_id]
        row = new_row.to_dict()
        if not previous.empty:
            old = previous.iloc[0].to_dict()
            row["created_at"] = old.get("created_at") or row.get("created_at")
        rows.append(row)

    keep = current[~current["asset_id"].astype(str).str.upper().isin({row["asset_id"] for row in rows})].copy()
    merged = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True, sort=False)
    save_dynamic_assets(merged)
    return merged


def update_dynamic_assets_from_query(query: str) -> dict:
    updates = parse_dynamic_asset_updates(query)
    current = load_dynamic_assets()
    if not updates:
        return {"updated": [], "missing": [], "history": []}

    now = datetime.now().isoformat(timespec="seconds")
    updated_rows: list[dict] = []
    missing: list[str] = []
    history_rows: list[dict] = []

    for update in updates:
        asset_id = str(update["asset_id"]).upper()
        idx = current.index[current["asset_id"].astype(str).str.upper() == asset_id].tolist()
        if not idx:
            missing.append(asset_id)
            continue

        row_idx = idx[0]
        previous_raw = current.loc[row_idx].to_dict()
        previous_scored = score_dynamic_asset(previous_raw)
        changed_fields: list[str] = []
        for field, value in update["fields"].items():
            if field in DYNAMIC_ASSET_COLUMNS:
                if field == "operator_notes":
                    old_note = _clean_text(current.at[row_idx, field], "")
                    new_note = _clean_text(value, "")
                    if old_note and new_note and new_note.lower() not in old_note.lower():
                        value = f"{old_note}; {new_note}"
                    elif old_note and not new_note:
                        value = old_note
                current.at[row_idx, field] = value
                changed_fields.append(field)
        current.at[row_idx, "updated_at"] = now
        current.at[row_idx, "source_query"] = str(query)
        new_raw = current.loc[row_idx].to_dict()
        new_scored = score_dynamic_asset(new_raw)
        updated_rows.append(new_scored)
        history_rows.append(
            {
                "event_id": str(uuid.uuid4()),
                "event_type": "update",
                "asset_id": asset_id,
                "changed_at": now,
                "changed_fields": json.dumps(changed_fields),
                "previous_score": previous_scored.get("hybrid_health_score"),
                "new_score": new_scored.get("hybrid_health_score"),
                "previous_priority": previous_scored.get("priority"),
                "new_priority": new_scored.get("priority"),
                "previous_risk_band": previous_scored.get("risk_band"),
                "new_risk_band": new_scored.get("risk_band"),
                "source_query": str(query),
                "previous_record": json.dumps(_json_clean(previous_scored)),
                "new_record": json.dumps(_json_clean(new_scored)),
            }
        )

    if updated_rows:
        save_dynamic_assets(current)
        append_dynamic_asset_history(history_rows)
    return {"updated": updated_rows, "missing": missing, "history": history_rows}


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
    area = _clean_text(r.get("area"), "Unspecified Area")
    criticality = _clean_text(r.get("criticality"), "Medium").title()

    raw_temp = _num(r.get("temperature"), np.nan)
    raw_vib = _num(r.get("vibration"), np.nan)
    raw_current = _num(r.get("current"), np.nan)
    raw_pressure = _num(r.get("pressure"), np.nan)
    raw_rpm = _num(r.get("rpm"), 1480.0)
    raw_alarms = _num(r.get("alarm_count"), 0.0)
    operator_notes = _clean_text(r.get("operator_notes"), "")
    notes = operator_notes.lower()

    temp = 0.0 if np.isnan(raw_temp) else raw_temp
    vib = 0.0 if np.isnan(raw_vib) else raw_vib
    current = 0.0 if np.isnan(raw_current) else raw_current
    pressure = 8.0 if np.isnan(raw_pressure) else raw_pressure
    alarms = 0.0 if np.isnan(raw_alarms) else raw_alarms
    rpm = 1480.0 if np.isnan(raw_rpm) else raw_rpm

    missing = [
        field
        for field, value in [
            ("temperature", raw_temp),
            ("vibration", raw_vib),
            ("current", raw_current),
            ("pressure", raw_pressure),
        ]
        if np.isnan(value)
    ]

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

    if "blast furnace" in (typ + " " + str(area).lower()) and vib >= 6.5 and temp >= 80:
        score += 20
    if any(word in typ for word in ["blower", "fan", "compressor"]) and vib >= 6 and current >= 80:
        score += 12
    if "hydraulic" in typ and pressure <= 6:
        score += 10
    if "pump" in typ and pressure <= 6:
        score += 10
    if "caster" in (typ + " " + str(area).lower()) and vib >= 6:
        score += 8
    if "descaler" in typ and pressure <= 6:
        score += 10
    if any(term in notes for term in ["cavitation", "suction noise"]) and any(term in typ for term in ["pump", "descaler"]):
        score += 25
        score = max(score, 58)
    if any(term in notes for term in ["loud noise", "abnormal noise", "bearing noise", "rubbing", "chatter"]):
        score += 10
        score = max(score, 45)
    if any(term in notes for term in ["leak", "oil leak", "hydraulic leak"]):
        score += 15
    if any(term in notes for term in ["smoke", "sparking", "burning smell"]):
        score += 30
        score = max(score, 75)

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
        "area": area,
        "criticality": criticality,
        "criticality_score": criticality_score(criticality),
        "temperature": raw_temp,
        "vibration": raw_vib,
        "current": raw_current,
        "pressure": raw_pressure,
        "rpm": rpm,
        "alarm_count": int(alarms),
        "operator_notes": operator_notes,
        "temperature_for_scoring": temp,
        "vibration_for_scoring": vib,
        "current_for_scoring": current,
        "pressure_for_scoring": pressure,
        "missing_readings": ", ".join(missing),
        "provisional_scoring_note": "Missing readings held as unknown; neutral defaults used only for provisional risk scoring." if missing else "",
        "qualitative_risk_note": "Operator/field symptom used as risk override evidence." if operator_notes else "",
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
    if "caster" in typ:
        return "oscillator drive wear, mold friction instability, lubrication issue, alignment drift, or servo fault"
    return "abnormal operating condition inferred from current readings, alarm state, criticality, and equipment class"


def dynamic_actions(asset_type: str) -> list[str]:
    typ = str(asset_type).lower()
    if "blower" in typ or "fan" in typ:
        return [
            "Verify blower/fan vibration spectrum, bearing temperature, motor current balance, damper position, and foundation bolts.",
            "Check impeller fouling, duct restriction, lubrication condition, coupling alignment, and standby equipment availability.",
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
    if "caster" in typ:
        return [
            "Inspect mold oscillator drive, servo response, lubrication condition, alignment, vibration spectrum, and breakout-prevention interlocks.",
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
            ("Blower/fan bearing set", 1, 10, "Critical", 95000),
            ("Coupling insert/alignment kit", 2, 3, "High", 18000),
        ]
    elif "pump" in typ:
        parts = [
            ("Pump seal kit", 1, 5, "High", 28000),
            ("Pump bearing set", 1, 7, "High", 42000),
        ]
    elif "caster" in typ:
        parts = [
            ("Mold oscillator bearing kit", 1, 14, "Critical", 135000),
            ("Servo feedback sensor", 1, 9, "High", 52000),
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
