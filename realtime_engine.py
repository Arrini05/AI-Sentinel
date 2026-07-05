# ===== realtime_engine.py =====

"""
Thin wrapper for backwards compatibility.

All real functionality now lives in live_traffic.LiveTrafficEngine.
This module re-exports that class so any code that imports 
realtime_engine.LiveTrafficEngine continues to work.

Version: 2.0.0 (legacy wrapper)
"""

from live_traffic import LiveTrafficEngine  # noqa: F401

__all__ = ["LiveTrafficEngine"]