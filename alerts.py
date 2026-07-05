# ===== alerts.py (FULLY CORRECTED) =====

import os
import json
import logging
import random
import threading
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Alert Data Structure
# ─────────────────────────────────────────────────────────────
def _generate_id() -> str:
    import uuid
    return str(uuid.uuid4())[:8].upper()


@dataclass
class Alert:
    timestamp: str
    source_ip: str
    destination: str
    protocol: str
    prediction: int          # 0 = normal, 1 = attack
    confidence: float        # 0.0 – 1.0
    severity: str           # "Low" | "Medium" | "High" | "Critical"
    model_name: str
    alert_id: str = field(default_factory=_generate_id)
    # Enrichment fields
    attack_type: str = "Unknown"
    country: str = "Unknown"
    flag: str = "🌐"
    isp: str = "Unknown"
    is_ddos: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _severity_from_confidence(confidence: float) -> str:
    if confidence >= 0.90:
        return "Critical"
    elif confidence >= 0.75:
        return "High"
    elif confidence >= 0.55:
        return "Medium"
    return "Low"


def _fake_ip() -> str:  # FIX: Corrected closing parens
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


# ─────────────────────────────────────────────────────────────
# Alert Manager
# ─────────────────────────────────────────────────────────────
class AlertManager:
    """
    Thread-safe alert manager.
    - Triggers alerts for any sample predicted as attack (label == 1).
    - Logs each alert to a plain-text log file and a JSON lines file.
    - Maintains an in-memory list of the most recent alerts.
    """

    def __init__(
        self,
        log_path: str = config.ALERT_LOG_PATH,
        json_path: str = config.ALERT_JSON_PATH,
        max_memory: int = 1000,
    ):
        self.log_path = log_path
        self.json_path = json_path
        self.max_memory = max_memory
        self._alerts: list = []
        self._lock = threading.Lock()
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        logger.info("[AlertManager] Ready. Log → %s", log_path)

    # ── Public API ─────────────────────────────────────────────
    def process_predictions(
        self,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray],
        model_name: str,
        metadata: Optional[list] = None,
    ) -> list:
        """Inspect model output and fire alerts for malicious samples."""
        if probabilities is None:
            probabilities = predictions.astype(float)
        
        fired = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if pred == 1:
                meta = (metadata[i] if metadata and i < len(metadata) else {})
                alert = self._create_alert(pred, prob, model_name, meta)
                self._store(alert)
                fired.append(alert)
        
        if fired:
            logger.warning("[AlertManager] %d alert(s) fired in this batch.", len(fired))
        
        return fired

    def store_alert(self, alert: Alert) -> None:
        """Directly store a pre-built Alert object."""
        self._store(alert)

    def get_recent_alerts(self, n: int = 50) -> list:
        with self._lock:
            return list(self._alerts[-n:])

    def get_all_alerts(self) -> list:
        with self._lock:
            return list(self._alerts)

    def clear_memory(self) -> None:
        with self._lock:
            self._alerts.clear()
        logger.info("[AlertManager] In-memory buffer cleared.")

    @property
    def alert_count(self) -> int:
        with self._lock:
            return len(self._alerts)

    def load_from_json(self) -> None:
        """Reload persisted alerts from the JSON-lines file into memory."""
        if not os.path.isfile(self.json_path):
            return
        
        loaded = 0
        with open(self.json_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    d.setdefault("attack_type", "Unknown")
                    d.setdefault("country", "Unknown")
                    d.setdefault("flag", "🌐")
                    d.setdefault("isp", "Unknown")
                    d.setdefault("is_ddos", False)
                    alert = Alert(**d)
                    with self._lock:
                        self._alerts.append(alert)
                    loaded += 1
                except Exception as exc:
                    logger.warning("[AlertManager] Skipping malformed JSON line: %s", exc)
        
        logger.info("[AlertManager] Loaded %d alerts from %s", loaded, self.json_path)

    # ── Internals ─────────────────────────────────────────────
    def _create_alert(self, prediction, confidence, model_name, meta) -> Alert:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        severity = _severity_from_confidence(float(confidence))
        
        return Alert(
            timestamp=ts,
            source_ip=meta.get("source_ip", _fake_ip()),
            destination=meta.get("destination", _fake_domain()),
            protocol=meta.get("protocol", "TCP"),
            prediction=int(prediction),
            confidence=round(float(confidence), 4),
            severity=meta.get("severity", severity),
            model_name=model_name,
            attack_type=meta.get("attack_type", "Unknown"),
            country=meta.get("country", "Unknown"),
            flag=meta.get("flag", "🌐"),
            isp=meta.get("isp", "Unknown"),
            is_ddos=meta.get("is_ddos", False),
        )

    def _store(self, alert: Alert) -> None:
        """Persist to files and append to memory buffer (thread-safe)."""
        self._write_text_log(alert)
        self._write_json_log(alert)
        
        with self._lock:
            self._alerts.append(alert)
            if len(self._alerts) > self.max_memory:
                self._alerts.pop(0)

    def _write_text_log(self, alert: Alert) -> None:
        ddos_tag = " [DDoS]" if alert.is_ddos else ""
        line = (
            f"[{alert.timestamp}] ALERT#{alert.alert_id} | "
            f"{alert.severity:8s} | {alert.attack_type}{ddos_tag} | "
            f"model={alert.model_name} | "
            f"src={alert.source_ip} ({alert.flag} {alert.country}) "
            f"→ {alert.destination} | "
            f"proto={alert.protocol} | conf={alert.confidence:.2%}\n"
        )
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError as exc:
            logger.error("[AlertManager] Could not write text log: %s", exc)

    def _write_json_log(self, alert: Alert) -> None:
        try:
            with open(self.json_path, "a", encoding="utf-8") as fh:
                # FIX: Added ensure_ascii=False for proper emoji handling
                fh.write(json.dumps(alert.to_dict(), ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("[AlertManager] Could not write JSON log: %s", exc)


# ─────────────────────────────────────────────────────────────
# Fake Metadata Generators
# ─────────────────────────────────────────────────────────────
_FAKE_DOMAINS = [
    "phishing-login.com", "botnet-control.net", "malware-host.org",
    "ddos-attack.site", "ransomware-pay.net", "exploit-kit.biz",
]
_PROTOCOLS = ["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "DNS"]


def _fake_domain() -> str:
    return random.choice(_FAKE_DOMAINS)


def generate_fake_metadata(n: int) -> list:
    """Generate n dicts of simulated network metadata."""
    return [
        {
            "source_ip": _fake_ip(),
            "destination": _fake_domain(),
            "protocol": random.choice(_PROTOCOLS),
        }
        for _ in range(n)
    ]


# ─────────────────────────────────────────────────────────────
# Enrich Alert with Threat-Intel Data
# ─────────────────────────────────────────────────────────────
def enrich_alert(alert: Alert, geo: dict) -> Alert:
    """Apply GeoIP / threat-intel enrichment to an existing Alert in place."""
    if geo:
        alert.country = geo.get("country", alert.country)
        alert.flag = geo.get("flag", alert.flag)
        alert.isp = geo.get("isp", alert.isp)
    return alert