# ===== threat_intel.py (FULLY CORRECTED) =====

"""
Threat Intelligence Enrichment for AI Sentinel.

Features
--------
-  GeoIP lookup via ip-api.com (free, no API key needed)
-  ISP / org / AS enrichment
-  Local known-malicious IP set
-  LRU cache to avoid hammering the API
-  Graceful fallback when offline
"""

from __future__ import annotations

import ipaddress
import logging
import time
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Geo-IP Lookup (Cached)
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=config.GEOIP_CACHE_SIZE)
def lookup_ip(ip: str) -> Dict:
    """
    Return enrichment dict for an IP address.
    
    Keys: ip, country, region, city, isp, org, asn, is_known_malicious, flag_emoji
    """
    # Private/loopback fast-path
    if _is_private(ip):
        return _private_result(ip)

    # Known-malicious check
    known_bad = ip in config.KNOWN_MALICIOUS_IPS

    # API call
    try:
        import urllib.request
        import json

        url = config.GEOIP_API_URL.format(ip=ip)
        req = urllib.request.Request(url, headers={"User-Agent": "AI-Sentinel/1.0"})

        with urllib.request.urlopen(req, timeout=config.GEOIP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())

        if data.get("status") != "success":
            raise ValueError("API status != success")

        return {
            "ip": ip,
            "country": data.get("country", "Unknown"),
            "region": data.get("regionName", ""),
            "city": data.get("city", ""),
            "isp": data.get("isp", "Unknown ISP"),
            "org": data.get("org", ""),
            "asn": data.get("as", ""),
            "is_known_malicious": known_bad,
            "flag_emoji": _country_flag(data.get("country", "")),
        }

    except Exception as exc:
        logger.debug("[ThreatIntel] GeoIP lookup failed for %s: %s", ip, exc)
        return _unknown_result(ip, known_bad)


def enrich_alert(alert_dict: Dict) -> Dict:
    """Add threat-intel fields to an alert dict."""
    ip = alert_dict.get("source_ip", "")
    geo = lookup_ip(ip)
    alert_dict["geo"] = geo
    alert_dict["threat_summary"] = _build_summary(alert_dict, geo)
    return alert_dict


def enrich_alerts(alerts: List[Dict]) -> List[Dict]:
    """Batch enrich alert dicts."""
    return [enrich_alert(a) for a in alerts]


# ──────────────────────────────────────────────────────────────
# Summary Builder
# ──────────────────────────────────────────────────────────────

def _build_summary(alert: Dict, geo: Dict) -> str:
    """Build human-readable threat summary."""
    parts = []

    attack_type = alert.get("attack_type", "Attack")
    confidence = alert.get("confidence", 0.0)
    severity = alert.get("severity", "")
    src_ip = alert.get("source_ip", "")
    dst = alert.get("destination", "")
    protocol = alert.get("protocol", "")

    parts.append(f"🚨 {attack_type} detected [{severity}]")
    parts.append(f"   Confidence : {confidence:.0%}")
    parts.append(f"   Source IP  : {src_ip}")

    country = geo.get("country", "")
    city = geo.get("city", "")
    isp = geo.get("isp", "")
    flag = geo.get("flag_emoji", "")

    if country:
        loc = f"{flag} {city}, {country}".strip(" ,")
        parts.append(f"   Location   : {loc}")
    if isp:
        parts.append(f"   ISP        : {isp}")

    if geo.get("is_known_malicious"):
        parts.append("   ⚠️  IP is on known-malicious list!")

    parts.append(f"   Destination: {dst} [{protocol}]")
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _is_private(ip: str) -> bool:
    """Check for RFC-1918 / loopback / link-local."""
    try:
        obj = ipaddress.ip_address(ip)
        return obj.is_private or obj.is_loopback or obj.is_link_local
    except ValueError:
        return False


def _private_result(ip: str) -> Dict:
    return {
        "ip": ip,
        "country": "Private Network",
        "region": "",
        "city": "",
        "isp": "Internal",
        "org": "",
        "asn": "",
        "is_known_malicious": False,
        "flag_emoji": "🏠",
    }


def _unknown_result(ip: str, known_bad: bool) -> Dict:
    return {
        "ip": ip,
        "country": "Unknown",
        "region": "",
        "city": "",
        "isp": "Unknown",
        "org": "",
        "asn": "",
        "is_known_malicious": known_bad,
        "flag_emoji": "🌐",
    }


def _country_flag(country_name: str) -> str:
    """Flag emoji mapping."""
    FLAGS = {
        "United States": "🇺🇸",
        "China": "🇨🇳",
        "Russia": "🇷🇺",
        "Germany": "🇩🇪",
        "United Kingdom": "🇬🇧",
        "France": "🇫🇷",
        "Brazil": "🇧🇷",
        "India": "🇮🇳",
        "Canada": "🇨🇦",
        "Netherlands": "🇳🇱",
        "South Korea": "🇰🇷",
        "Japan": "🇯🇵",
        "Australia": "🇦🇺",
        "Ukraine": "🇺🇦",
    }
    return FLAGS.get(country_name, "🌐")


# ──────────────────────────────────────────────────────────────
# DDoS / Flood Detector
# ──────────────────────────────────────────────────────────────

class FloodDetector:
    """Track per-IP packet counts in sliding time window."""

    def __init__(self, window_sec: int = config.DDOS_WINDOW_SEC,
                 threshold: int = config.DDOS_PACKET_THRESH):
        self.window_sec = window_sec
        self.threshold = threshold
        self._buckets: Dict[str, list] = {}

    def record(self, ip: str) -> bool:
        """Record packet, return True if DDoS threshold breached."""
        now = time.monotonic()
        bucket = self._buckets.setdefault(ip, [])

        # Evict old entries
        cutoff = now - self.window_sec
        self._buckets[ip] = [t for t in bucket if t >= cutoff]
        self._buckets[ip].append(now)

        count = len(self._buckets[ip])
        if count >= self.threshold:
            logger.warning(
                "[FloodDetector] DDoS: %s (%d / %ds)",
                ip, count, self.window_sec
            )
            return True
        return False

    def get_top_ips(self, n: int = 10) -> List[Tuple[str, int]]:
        """Return top-n IPs by packet count."""
        counts = {ip: len(ts) for ip, ts in self._buckets.items()}
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]

    def reset(self) -> None:
        """Clear all buckets."""
        self._buckets.clear()