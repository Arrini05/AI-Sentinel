# ===== live_traffic.py (FULLY CORRECTED) =====

import logging
import os
import threading
import time
import random
from collections import deque
from datetime import datetime
from typing import Optional, Dict, Any, List

import joblib
import pandas as pd
import numpy as np

# Try to import scapy (optional - for real capture)
try:
    from scapy.all import sniff, IP, TCP, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

import config
from preprocessing import load_preprocessor, load_selector
from models import load_model, predict, predict_proba
from alerts import AlertManager, Alert
from attack_classifier import AttackTypeClassifier
from smart_model_selector import SmartModelSelector

logger = logging.getLogger(__name__)

# Feature columns for raw packet capture (real NIC / PCAP modes only).
# NOTE: these do NOT match the NSL-KDD/UNSW schema that `preprocessor` was
# fit on, so they are handled by a separate lightweight model
# (see train_live_model.py) rather than being pushed through `preprocessor`.
FEATURE_COLUMNS = [
    "packet_length", "src_port", "dst_port",
    "protocol", "ttl", "tcp_flags"
]

# Where train_live_model.py saves its artifacts
LIVE_MODEL_PATH = os.path.join(config.MODEL_DIR, "live_rf_model.joblib")
LIVE_SCALER_PATH = os.path.join(config.MODEL_DIR, "live_scaler.joblib")


class LiveTrafficEngine:
    """
    Live traffic processing engine.
    Supports multiple modes: synthetic, replay, real NIC, PCAP

    IMPORTANT (architecture note):
    `X_test` (used in "replay" mode) is already fully preprocessed --
    scaled, balanced, and feature-selected by `preprocessing.preprocess()`.
    It must NOT be pushed through `preprocessor`/`selector` again. Those
    are only meaningful for raw, dataset-shaped rows.

    Raw packet captures ("real" / "pcap" modes) have a totally different,
    much smaller schema (6 basic fields) than the NSL-KDD/UNSW datasets the
    main models were trained on, so they are routed through a dedicated,
    separately-trained lightweight model (see train_live_model.py) instead.
    "synthetic" mode samples + jitters real X_test rows so it can exercise
    the real models exactly like replay mode does, without needing any of
    the above conversions.
    """

    def __init__(
        self,
        models: Optional[dict] = None,
        preprocessor=None,
        selector=None,
        alert_manager: Optional[AlertManager] = None,
        attack_clf: Optional[AttackTypeClassifier] = None,
        batch_size: int = 32,
        interval_sec: float = 1.0,
        attack_ratio: float = 0.35,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        use_real_capture: bool = False,
        interface: Optional[str] = None,
        pcap_path: Optional[str] = None,
        mode_override: str = "synthetic",
    ):
        self.models = models or {}

        # Get primary model
        self.model = None
        self._primary_model_name = None
        if self.models:
            self._primary_model_name = next(iter(self.models.keys()))
            self.model = self.models[self._primary_model_name]

        # Smart selector
        self.smart_selector: Optional[SmartModelSelector] = None
        if self.models:
            try:
                self.smart_selector = SmartModelSelector(self.models)
            except Exception as e:
                logger.warning(f"Smart selector init failed: {e}")

        self.preprocessor = preprocessor
        self.selector = selector
        self.alert_manager = alert_manager
        self.attack_clf = attack_clf

        self.batch_size = batch_size
        self.interval_sec = interval_sec
        self.attack_ratio = attack_ratio

        # Replay data (ALREADY preprocessed -- see class docstring)
        self.X_test = X_test
        self.y_test = y_test
        self._replay_idx = 0

        # Random generator for synthetic mode (FIX: was referenced as
        # self.rng but never created, which crashed the synthetic loop)
        self.rng = np.random.default_rng()

        # Capture settings
        self.use_real_capture = use_real_capture
        self.interface = interface
        self.pcap_path = pcap_path
        self.mode = mode_override

        self.is_running = False

        # Dedicated lightweight model for raw packet capture (real/pcap).
        # Lazily loaded; absence is reported once via a clear log message
        # instead of silently dropping every packet.
        self.live_model = None
        self.live_scaler = None
        self._load_live_capture_model()
        self._warned_no_live_model = False

        # Statistics
        self.stats = {
            "total_packets": 0,
            "total_attacks": 0,
            "packets_per_sec": 0.0,
            "current_model": self._primary_model_name or "random_forest",
            "model_reason": "manual",
        }
        self._stat_lock = threading.Lock()

        # Results queue
        self.result_queue = deque(maxlen=5000)

        # Timing for PPS calculation
        self._last_time = time.time()
        self._packet_count = 0

    # ─────────────────────────────────────────────────────────────
    # Engine Control
    # ─────────────────────────────────────────────────────────────

    def start_stream(self):
        """Start the traffic engine."""
        if self.is_running:
            return

        self.is_running = True
        thread = threading.Thread(target=self._run_loop, daemon=True)
        thread.start()
        logger.info("[LiveTraffic] Engine started in mode: %s", self.mode)

    def stop_stream(self):
        """Stop the traffic engine."""
        self.is_running = False
        logger.info("[LiveTraffic] Engine stopped")

    def drain_results(self, max_items: int = 50) -> List[Dict]:
        """Drain processed results from the queue."""
        results = []
        while self.result_queue and len(results) < max_items:
            results.append(self.result_queue.popleft())
        return results

    def get_stats(self) -> Dict:
        """Get current engine statistics."""
        return self.stats.copy()

    def set_primary_model(self, name: str) -> None:
        """
        Switch which model the engine uses for direct (non-smart-selector)
        prediction. FIX: previously, changing the "Primary Model" dropdown
        in the dashboard never actually changed what the live engine ran --
        `self.model` was fixed once at engine-creation time.
        """
        if name in self.models:
            self.model = self.models[name]
            self._primary_model_name = name
            with self._stat_lock:
                self.stats["current_model"] = name
                self.stats["model_reason"] = "user selection"

    # ─────────────────────────────────────────────────────────────
    # Live-capture model (raw packet fields -- real NIC / PCAP only)
    # ─────────────────────────────────────────────────────────────

    def _load_live_capture_model(self) -> None:
        """
        Load the lightweight model trained by train_live_model.py on raw
        packet-capture features. This is intentionally separate from the
        NSL-KDD/UNSW `preprocessor`, whose schema (41 named dataset columns)
        has nothing to do with what a live packet capture can produce.
        """
        try:
            if os.path.isfile(LIVE_MODEL_PATH) and os.path.isfile(LIVE_SCALER_PATH):
                self.live_model = joblib.load(LIVE_MODEL_PATH)
                self.live_scaler = joblib.load(LIVE_SCALER_PATH)
                logger.info("[LiveTraffic] Loaded live-capture model ← %s", LIVE_MODEL_PATH)
        except Exception as e:
            logger.warning("[LiveTraffic] Could not load live-capture model: %s", e)
            self.live_model = None
            self.live_scaler = None

    # ─────────────────────────────────────────────────────────────
    # Feature Extraction (raw packets)
    # ─────────────────────────────────────────────────────────────

    def extract_features(self, packet) -> Optional[Dict]:
        """Extract raw features from a scapy packet."""
        try:
            if IP not in packet:
                return None

            ip = packet[IP]
            # FIX: ip.proto is a raw integer (6=TCP, 17=UDP, 1=ICMP, ...).
            # Used directly, this broke the Protocol Distribution chart in the
            # Overview tab — bars showed numeric codes instead of names, and
            # didn't line up with PROTOCOL_COLORS / dashboard expectations.
            # Map to the same string labels used everywhere else in the app.
            _PROTO_MAP = {1: "ICMP", 6: "TCP", 17: "UDP"}
            protocol = _PROTO_MAP.get(ip.proto, "Other")
            ttl = ip.ttl
            packet_length = len(packet)

            src_port = dst_port = 0
            tcp_flags = 0

            if TCP in packet:
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
                tcp_flags = int(packet[TCP].flags)
            elif UDP in packet:
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport

            # Map packet to a meaningful service label for the Protocol
            # Distribution chart. Use BOTH dst_port and src_port since
            # response packets have the well-known port as the SOURCE.
            _PORT_SERVICE = {
                # Web
                80: "HTTP", 443: "HTTPS", 8080: "HTTP", 8443: "HTTPS",
                8000: "HTTP", 8888: "HTTP", 3000: "HTTP", 5000: "HTTP",
                # DNS
                53: "DNS",
                # Email
                25: "SMTP", 465: "SMTP", 587: "SMTP",
                110: "POP3", 995: "POP3",
                143: "IMAP", 993: "IMAP",
                # File transfer
                20: "FTP", 21: "FTP",
                # Remote access
                22: "SSH", 23: "Telnet", 3389: "RDP",
                # Windows networking
                445: "SMB", 139: "SMB", 135: "RPC", 137: "NetBIOS",
                138: "NetBIOS",
                # Database
                3306: "MySQL", 5432: "PostgreSQL", 1433: "MSSQL",
                # Other common
                67: "DHCP", 68: "DHCP", 123: "NTP", 161: "SNMP",
                162: "SNMP", 389: "LDAP", 636: "LDAPS",
                1194: "VPN", 1723: "PPTP", 500: "IPSec",
                # Streaming / gaming
                554: "RTSP", 1935: "RTMP",
            }

            # Check dst_port first, then src_port (catches response packets)
            service_label = (
                _PORT_SERVICE.get(dst_port) or
                _PORT_SERVICE.get(src_port) or
                # Unknown port — label by IP protocol so chart shows TCP/UDP/ICMP
                # rather than collapsing everything into one "TCP" bar
                (f"TCP:{dst_port}" if protocol == "TCP" and dst_port > 0 and dst_port < 1024
                 else protocol)
            )

            return {
                "packet_length": packet_length,
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": protocol,          # string label for display
                "proto_num": ip.proto,         # raw int for model input
                "service_label": service_label,
                "ttl": ttl,
                "tcp_flags": tcp_flags,
            }
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return None

    def _synthetic_row(self) -> np.ndarray:
        """
        Generate a realistic, already-model-ready feature row for the
        synthetic demo mode.

        FIX: the old implementation built 6 raw packet-style fields and
        pushed them through the NSL-KDD `preprocessor`, which expects ~41
        named dataset columns -- that always raised an exception that was
        silently swallowed. Instead, we sample (and lightly jitter) a real
        row from `X_test` so synthetic mode exercises the *actual* trained
        model exactly the way replay mode does, with zero extra plumbing.
        """
        if self.X_test is not None and len(self.X_test) > 0:
            idx = int(self.rng.integers(0, len(self.X_test)))
            row = np.asarray(self.X_test[idx], dtype=float).copy()
            # Light Gaussian jitter so repeated draws aren't identical
            row = row + self.rng.normal(0, 0.05, size=row.shape)
            is_attack = self.rng.random() < self.attack_ratio
            if not is_attack:
                # Nudge gently toward the "normal" end for variety
                row = row * 0.9
            return row

        # Fallback if no test data is available: a unit-normal vector sized
        # to whatever the active model expects.
        n_features = getattr(self.model, "n_features_in_", 20) or 20
        return self.rng.normal(0, 1, size=n_features)

    # ─────────────────────────────────────────────────────────────
    # Packet Processing
    # ─────────────────────────────────────────────────────────────

    def _predict_row(self, X_row: np.ndarray, true_label: Optional[int] = None) -> Optional[Dict]:
        """
        Run inference on an already-model-ready feature row (replay /
        synthetic modes) and push the result onto the queue.
        """
        if not self.model and not self.models:
            return None

        try:
            X = np.asarray(X_row, dtype=float).reshape(1, -1)

            start_time = time.perf_counter()

            if self.smart_selector:
                preds, probs, model_name, reason = self.smart_selector.predict(X)
            else:
                preds = self.model.predict(X)
                probs = predict_proba(self.model, X)
                model_name = self._primary_model_name or list(self.models.keys())[0]
                reason = "manual"

            inference_time = (time.perf_counter() - start_time) * 1000

            prediction = int(preds[0])

            with self._stat_lock:
                self.stats["current_model"] = model_name
                self.stats["model_reason"] = reason

            probs = np.asarray(probs)
            if probs.ndim == 2:
                probability = float(probs[0, 1]) if probs.shape[1] > 1 else float(probs[0, 0])
            else:
                probability = float(probs[0])

            attack_type = "Normal"
            if prediction == 1:
                attack_type = "Attack"
                if self.attack_clf:
                    try:
                        clf_result = self.attack_clf.predict(X)[0]
                        attack_type = clf_result.get("attack_type", "Attack")
                    except Exception:
                        attack_type = "Attack"

            result = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_ip": f"{np.random.randint(1,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(1,255)}",
                "destination": "192.168.1.1",
                "protocol": random.choice(["TCP", "UDP", "HTTPS", "DNS"]),
                "confidence": probability,
                "binary_pred": prediction,
                "attack_type": attack_type,
                "country": "Unknown",
                "flag": "🌐",
                "inference_ms": round(inference_time, 3),
            }
            if true_label is not None:
                result["true_label"] = int(true_label)

            self.result_queue.append(result)

            with self._stat_lock:
                self.stats["total_packets"] += 1
                if prediction == 1:
                    self.stats["total_attacks"] += 1

                self._packet_count += 1
                current_time = time.time()

                if current_time - self._last_time >= 1:
                    self.stats["packets_per_sec"] = (
                        self._packet_count / (current_time - self._last_time)
                    )
                    self._packet_count = 0
                    self._last_time = current_time

            return result

        except Exception as e:
            # FIX: was logger.debug() (invisible at the dashboard's WARNING
            # level), which is exactly why every packet vanished silently.
            logger.warning(f"[LiveTraffic] Inference error: {e}")
            return None

    def process_raw_packet_features(self, features: Dict) -> Optional[Dict]:
        """
        Process raw packet-capture features (real NIC / PCAP modes) using
        the dedicated lightweight live-capture model. These features do NOT
        match the NSL-KDD/UNSW preprocessor's schema, so that preprocessor
        is intentionally not used here.
        """
        if self.live_model is None or self.live_scaler is None:
            if not self._warned_no_live_model:
                logger.warning(
                    "[LiveTraffic] No live-capture model found at %s. "
                    "Run `python train_live_model.py` (after generating "
                    "data/live_dataset.csv) to enable Real NIC / PCAP modes, "
                    "or switch to Dataset Replay / Synthetic mode.",
                    LIVE_MODEL_PATH,
                )
                self._warned_no_live_model = True
            return None

        try:
            X_raw = np.array([[
                features.get("packet_length", 0),
                features.get("src_port", 0),
                features.get("dst_port", 0),
                features.get("proto_num", 6),   # numeric IP protocol for model
                features.get("ttl", 0),
                features.get("tcp_flags", 0),
            ]], dtype=float)
            X = self.live_scaler.transform(X_raw)

            start_time = time.perf_counter()
            preds = self.live_model.predict(X)
            probs = predict_proba(self.live_model, X)
            inference_time = (time.perf_counter() - start_time) * 1000

            prediction = int(preds[0])
            probs = np.asarray(probs)
            probability = float(probs[0, 1]) if probs.ndim == 2 and probs.shape[1] > 1 else float(np.ravel(probs)[0])

            with self._stat_lock:
                self.stats["current_model"] = "live_rf_model"
                self.stats["model_reason"] = "raw packet capture"

            attack_type = "Attack" if prediction == 1 else "Normal"

            # FIX: was hardcoded "TCP" for every single packet, regardless of
            # what the packet actually was — this is why the Protocol
            # Distribution chart only ever showed one flat TCP bar. Use the
            # service-aware label (HTTPS/HTTP/DNS/TCP/UDP/...) computed in
            # extract_features() so normal browsing still shows variety.
            protocol_label = features.get("service_label") or features.get("protocol", "Other")
            if isinstance(protocol_label, (int, float)):
                protocol_label = {1: "ICMP", 6: "TCP", 17: "UDP"}.get(int(protocol_label), "Other")

            result = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_ip": f"{np.random.randint(1,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(1,255)}",
                "destination": "192.168.1.1",
                "protocol": protocol_label,
                "confidence": probability,
                "binary_pred": prediction,
                "attack_type": attack_type,
                "country": "Unknown",
                "flag": "🌐",
                "inference_ms": round(inference_time, 3),
            }
            self.result_queue.append(result)

            with self._stat_lock:
                self.stats["total_packets"] += 1
                if prediction == 1:
                    self.stats["total_attacks"] += 1
                self._packet_count += 1
                current_time = time.time()
                if current_time - self._last_time >= 1:
                    self.stats["packets_per_sec"] = (
                        self._packet_count / (current_time - self._last_time)
                    )
                    self._packet_count = 0
                    self._last_time = current_time

            return result

        except Exception as e:
            logger.warning(f"[LiveTraffic] Raw packet inference error: {e}")
            return None

    def process_packet(self, packet):
        """Process a scapy packet (real NIC / PCAP modes)."""
        features = self.extract_features(packet)
        if features:
            return self.process_raw_packet_features(features)

    # ─────────────────────────────────────────────────────────────
    # Main Loop
    # ─────────────────────────────────────────────────────────────

    def _run_loop(self):
        """Main engine loop."""
        logger.info("[LiveTraffic] Starting loop in mode: %s", self.mode)

        if self.mode == "replay":
            self._replay_loop()
        elif self.mode == "synthetic":
            self._synthetic_loop()
        elif self.mode == "real":
            self._capture_loop()
        elif self.mode == "pcap":
            self._pcap_loop()
        else:
            logger.warning(f"Unknown mode: {self.mode}")

    def _synthetic_loop(self):
        """Synthetic traffic generation loop (already-model-ready rows)."""
        while self.is_running:
            for _ in range(self.batch_size):
                row = self._synthetic_row()
                self._predict_row(row)

            time.sleep(self.interval_sec)

    def _replay_loop(self):
        """
        Replay test-set rows directly.

        FIX: `self.X_test` is already fully preprocessed (scaled, balanced,
        feature-selected) by `preprocessing.preprocess()`. The previous
        implementation re-wrapped a handful of these columns into a fake
        raw-packet dict and pushed it back through the NSL-KDD preprocessor
        a second time, which always raised (column-name mismatch) and was
        silently swallowed -- meaning replay mode, the *default* mode,
        never produced a single result. We now predict directly.
        """
        if self.X_test is None or self.y_test is None:
            logger.error("No X_test/y_test for replay mode")
            return

        X = self.X_test
        y = self.y_test
        n = len(X)

        while self.is_running:
            batch_end = min(self._replay_idx + self.batch_size, n)

            for i in range(self._replay_idx, batch_end):
                true_label = int(y[i]) if y is not None else None
                self._predict_row(X[i], true_label=true_label)

            self._replay_idx = batch_end % n

            time.sleep(self.interval_sec)

    def _capture_loop(self):
        """Real NIC capture loop."""
        if not SCAPY_AVAILABLE:
            logger.error("Scapy not available for capture mode")
            self.is_running = False
            return

        try:
            sniff(
                prn=self.process_packet,
                store=False,
                iface=self.interface,
                stop_filter=lambda _pkt: not self.is_running,
            )
        except Exception as e:
            logger.error(f"Capture error: {e}")
            self.is_running = False

    def _pcap_loop(self):
        """
        Replay packets from an uploaded PCAP file.

        FIX: this mode previously had no implementation at all -- `_run_loop`
        fell through to the "Unknown mode" branch and did nothing, despite
        the dashboard offering a full PCAP-upload UI for it.
        """
        if not SCAPY_AVAILABLE:
            logger.error("Scapy not available for PCAP replay")
            self.is_running = False
            return

        if not self.pcap_path or not os.path.isfile(self.pcap_path):
            logger.error("No valid PCAP file path provided")
            self.is_running = False
            return

        try:
            from scapy.all import rdpcap
            packets = rdpcap(self.pcap_path)
        except Exception as e:
            logger.error(f"Failed to read PCAP file: {e}")
            self.is_running = False
            return

        if not packets:
            logger.warning("PCAP file contains no packets")
            self.is_running = False
            return

        idx = 0
        n = len(packets)

        while self.is_running:
            batch_end = min(idx + self.batch_size, n)
            for i in range(idx, batch_end):
                self.process_packet(packets[i])
            idx = batch_end % n
            time.sleep(self.interval_sec)