# ===== automated_response.py  =====

"""
AI Sentinel - Automated Response System
Automatically respond to detected threats
"""

import os
import time
import json
import logging
import subprocess
import threading
from datetime import datetime
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("automated_response.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("automated_response")


class AutomatedResponse:
    def __init__(self, config=None):
        self.config = config or {}
        self.blocked_ips = set()
        self.blocked_ports = set()
        self.quarantined_ips = set()
        self.response_log = []
        self.auto_block = self.config.get("auto_block", False)
        self.auto_block_threshold = self.config.get("auto_block_threshold", 3)  # attacks before block
        self.ip_counts = defaultdict(int)
        self.enabled = True
        
    def log_response(self, action, target, details):
        """Log automated response action"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "target": target,
            "details": details
        }
        self.response_log.append(entry)
        logger.info(f"[{action}] {target} - {details}")
        
        # Write to SIEM log file
        self._write_siem_log(entry)
        
    def _write_siem_log(self, entry):
        """Write to SIEM-compatible log file"""
        try:
            with open("siem_log.json", "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"SIEM log write error: {e}")
            
    # ========================================
    # FIREWALL BLOCKING (Windows)
    # ========================================
    def block_ip_windows(self, ip):
        """Block IP using Windows Firewall"""
        try:
            rule_name = f"BLOCK_{ip.replace('.', '_')}_{int(time.time())}"
            
            # Windows Firewall rule
            cmd = [
                "netsh", "advfirewall", "firewall", "add", "rule",
                f"name={rule_name}",
                "dir=in",
                "action=block",
                f"remoteip={ip}",
                "enable=yes",
                "profile=any"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.blocked_ips.add(ip)
                self.log_response("BLOCK_IP", ip, "Windows Firewall rule added")
                return True
            else:
                logger.error(f"Failed to block IP {ip}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Block IP error: {e}")
            return False
            
    def unblock_ip_windows(self, ip):
        """Unblock IP using Windows Firewall"""
        try:
            # Find and remove rule
            result = subprocess.run(
                ["netsh", "advfirewall", "firewall", "show", "rule", "name=all"],
                capture_output=True,
                text=True
            )
            
            for line in result.stdout.split("\n"):
                if f"BLOCK_{ip.replace('.', '_')}" in line:
                    rule_name = line.split("=")[1].strip()
                    subprocess.run([
                        "netsh", "advfirewall", "firewall", "delete", "rule",
                        f"name={rule_name}"
                    ])
                    
            self.blocked_ips.discard(ip)
            self.log_response("UNBLOCK_IP", ip, "Windows Firewall rule removed")
            return True
            
        except Exception as e:
            logger.error(f"Unblock IP error: {e}")
            return False
            
    # ========================================
    # FIREWALL BLOCKING (Linux)
    # ========================================
    def block_ip_linux(self, ip):
        """Block IP using iptables (Linux)"""
        try:
            # Add iptables rule
            subprocess.run(["iptables", "-I", "INPUT", "-s", ip, "-j", "DROP"], check=True)
            subprocess.run(["iptables", "-I", "OUTPUT", "-d", ip, "-j", "DROP"], check=True)
            
            self.blocked_ips.add(ip)
            self.log_response("BLOCK_IP", ip, "iptables rule added (Linux)")
            return True
            
        except Exception as e:
            logger.error(f"Block IP error (Linux): {e}")
            return False
            
    def unblock_ip_linux(self, ip):
        """Unblock IP using iptables (Linux)"""
        try:
            # FIX: missing check=True — a failed `iptables -D` (no privileges,
            # rule not found, etc.) returned non-zero silently, so the IP was
            # removed from self.blocked_ips even though it was still blocked at
            # the OS level.
            subprocess.run(["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"], check=True)
            subprocess.run(["iptables", "-D", "OUTPUT", "-d", ip, "-j", "DROP"], check=True)
            
            self.blocked_ips.discard(ip)
            self.log_response("UNBLOCK_IP", ip, "iptables rule removed (Linux)")
            return True
            
        except Exception as e:
            logger.error(f"Unblock IP error (Linux): {e}")
            return False
            
    # ========================================
    # PORT BLOCKING
    # ========================================
    def block_port(self, port, protocol="tcp"):
        """Block a specific port"""
        try:
            if os.name == "nt":  # Windows
                cmd = [
                    "netsh", "advfirewall", "firewall", "add", "rule",
                    f"name=BLOCK_PORT_{port}_{protocol.upper()}",
                    "dir=in",
                    "action=block",
                    f"localport={port}",
                    f"protocol={protocol}"
                ]
                subprocess.run(cmd, check=True)
            else:  # Linux
                subprocess.run(["iptables", "-I", "INPUT", "-p", protocol, "--dport", str(port), "-j", "DROP"], check=True)
                
            self.blocked_ports.add(port)
            self.log_response("BLOCK_PORT", str(port), f"Protocol: {protocol}")
            return True
            
        except Exception as e:
            logger.error(f"Block port error: {e}")
            return False

    # ========================================
    # PORT UNBLOCKING
    # ========================================
    def unblock_port(self, port, protocol="tcp"):
        """Unblock a specific port"""
        try:
            if os.name == "nt":  # Windows
                cmd = [
                    "netsh", "advfirewall", "firewall", "delete", "rule",
                    f"name=BLOCK_PORT_{port}_{protocol.upper()}"
                ]
                subprocess.run(cmd, check=True)
            else:  # Linux
                subprocess.run(["iptables", "-D", "INPUT", "-p", protocol, "--dport", str(port), "-j", "DROP"])
            
            self.blocked_ports.discard(port)
            self.log_response("UNBLOCK_PORT", str(port), f"Protocol: {protocol}")
            return True
            
        except Exception as e:
            logger.error(f"Unblock port error: {e}")
            return False
            
    # ========================================
    # KILL CONNECTION
    # ========================================
    def kill_connection(self, ip, port=None):
        """Kill active connection to/from IP"""
        try:
            if os.name == "nt":  # Windows
                # Find and kill connection using netstat
                result = subprocess.run(
                    ["netstat", "-ano"],
                    capture_output=True,
                    text=True
                )
                
                for line in result.stdout.split("\n"):
                    if ip in line and (not port or str(port) in line):
                        parts = line.split()
                        if len(parts) >= 5:
                            pid = parts[-1]
                            subprocess.run(["taskkill", "/F", "/PID", pid], check=True)
                            
            self.log_response("KILL_CONNECTION", ip, f"Port: {port or 'all'}")
            return True
            
        except Exception as e:
            logger.error(f"Kill connection error: {e}")
            return False
            
    # ========================================
    # QUARANTINE HOST
    # ========================================
    def quarantine_host(self, ip, reason="Suspicious activity"):
        """Quarantine infected host (remove from network logically)"""
        # In production, this would:
        # 1. Notify network team
        # 2. Disable switch port
        # 3. Disable DHCP lease
        # 4. Log to SIEM
        
        self.quarantined_ips.add(ip)
        self.log_response("QUARANTINE", ip, reason)
        
        # Block the IP as well
        self.block_ip(ip)
        
    def lift_quarantine(self, ip):
        """Lift quarantine from host"""
        self.quarantined_ips.discard(ip)
        self.log_response("LIFT_QUARANTINE", ip, "Host cleared for return")
        
    # ========================================
    # RESET
    # ========================================
    def reset(self):
        """Reset all blocked entities"""
        self.blocked_ips.clear()
        self.blocked_ports.clear()
        self.quarantined_ips.clear()
        self.ip_counts.clear()
        self.response_log.clear()
        self.log_response("RESET", "All", "System reset")
        
    # ========================================
    # MAIN RESPONSE HANDLER
    # ========================================
    def handle_threat(self, alert):
        """
        Main threat handler - decides what action to take
        Returns: action taken
        """
        if not self.enabled:
            return "DISABLED"
            
        source_ip = alert.get("source_ip", "")
        attack_type = alert.get("attack_type", "Unknown")
        severity = alert.get("severity", "Medium")
        confidence = alert.get("confidence", 0)
        
        # Track IP counts for auto-blocking
        self.ip_counts[source_ip] += 1
        
        action_taken = None
        
        # Critical severity - auto block
        if severity == "Critical" and confidence > 0.9:
            if self.auto_block:
                self.block_ip(source_ip)
                action_taken = "AUTO_BLOCKED"
            else:
                self.log_response("ALERT_CRITICAL", source_ip, f"Attack: {attack_type}, Confidence: {confidence:.0%}")
                self.send_alert(alert)
                action_taken = "ALERTED"
        # High severity with repeated attacks
        elif severity == "High" and self.ip_counts[source_ip] >= self.auto_block_threshold:
            if self.auto_block:
                self.block_ip(source_ip)
                action_taken = "AUTO_BLOCKED"
            action_taken = "FLAGGED"
        # Medium - just log and alert
        else:
            self.log_response("DETECTED", source_ip, f"Attack: {attack_type}")
            action_taken = "LOGGED"
            
        return action_taken
        
    # ========================================
    # NOTIFICATIONS
    # ========================================
    def send_alert(self, alert):
        """Send email/Slack/etc alert (placeholder)"""
        # In production, integrate with:
        # - Email (SMTP)
        # - Slack (webhook)
        # - PagerDuty
        # - Microsoft Teams
        
        message = f"""
🚨 AI SENTINEL ALERT

Attack Type: {alert.get('attack_type')}
Source IP: {alert.get('source_ip')}
Severity: {alert.get('severity')}
Confidence: {alert.get('confidence'):.0%}
Time: {alert.get('timestamp')}
"""
        logger.warning(message)
        
    # ========================================
    # GET STATUS
    # ========================================
    def get_status(self):
        """Get automated response status"""
        return {
            "enabled": self.enabled,
            "auto_block": self.auto_block,
            "blocked_ips": len(self.blocked_ips),
            "blocked_ports": len(self.blocked_ports),
            "quarantined": len(self.quarantined_ips),
            "total_responses": len(self.response_log),
        }
        
    def get_response_log(self, limit=50):
        """Get recent response log"""
        return self.response_log[-limit:]
        
    def block_ip(self, ip):
        """Platform-independent block IP"""
        if os.name == "nt":
            return self.block_ip_windows(ip)
        else:
            return self.block_ip_linux(ip)
            
    def unblock_ip(self, ip):
        """Platform-independent unblock IP"""
        if os.name == "nt":
            return self.unblock_ip_windows(ip)
        else:
            return self.unblock_ip_linux(ip)


# Singleton instance
_response_system = None

def get_response_system(config=None):
    global _response_system
    if _response_system is None:
        _response_system = AutomatedResponse(config)
    return _response_system