# code : pipeline/alerts.py
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

ALERTS_PATH = Path(__file__).resolve().parents[2] / "alerts.json"


def load_alerts() -> list[dict[str, Any]]:
    try:
        data = json.loads(ALERTS_PATH.read_text(encoding="utf-8"))
        return [a for a in data if isinstance(a, dict) and a.get("sql")]
    except (OSError, json.JSONDecodeError):
        return []


def save_alert(name: str, sql: str, message: str) -> dict[str, Any]:
    """Enregistre une alerte (remplace une alerte du même nom)."""
    alerts = [a for a in load_alerts() if a.get("name", "").lower() != name.strip().lower()]
    alert = {
        "id": uuid.uuid4().hex[:8],
        "name": name.strip(),
        "sql": sql.strip(),
        "message": message.strip(),
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }
    alerts.append(alert)
    ALERTS_PATH.write_text(json.dumps(alerts, ensure_ascii=False, indent=2), encoding="utf-8")
    return alert