"""
debug_nic.py — Shows exactly what ports/protocols your NIC is seeing.
Run for 15 seconds then Ctrl+C:  .venv\Scripts\python debug_nic.py
"""
import sys, os, time
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scapy.all import sniff, IP, TCP, UDP, ICMP

port_counter   = Counter()
proto_counter  = Counter()
service_counter = Counter()

_PORT_SERVICE = {
    80: "HTTP", 443: "HTTPS", 8080: "HTTP", 8443: "HTTPS",
    53: "DNS", 25: "SMTP", 465: "SMTP", 587: "SMTP",
    110: "POP3", 143: "IMAP", 993: "IMAP", 995: "POP3",
    20: "FTP", 21: "FTP", 22: "SSH", 23: "Telnet",
    3389: "RDP", 445: "SMB", 139: "SMB", 135: "RPC",
    67: "DHCP", 68: "DHCP", 123: "NTP", 161: "SNMP",
    3306: "MySQL", 5432: "PostgreSQL", 1433: "MSSQL",
    3000: "HTTP", 5000: "HTTP", 8000: "HTTP", 8888: "HTTP",
}

def cb(pkt):
    if IP not in pkt:
        return
    ip = pkt[IP]
    _PROTO_MAP = {1: "ICMP", 6: "TCP", 17: "UDP"}
    proto = _PROTO_MAP.get(ip.proto, f"proto_{ip.proto}")
    proto_counter[proto] += 1

    dst_port = src_port = 0
    if TCP in pkt:
        dst_port = pkt[TCP].dport
        src_port = pkt[TCP].sport
    elif UDP in pkt:
        dst_port = pkt[UDP].dport
        src_port = pkt[UDP].sport

    if dst_port: port_counter[f"dst:{dst_port}"] += 1
    if src_port: port_counter[f"src:{src_port}"] += 1

    svc = (_PORT_SERVICE.get(dst_port) or
           _PORT_SERVICE.get(src_port) or proto)
    service_counter[svc] += 1

print("Capturing 15 seconds of traffic… browse normally, open some websites.")
print("Ctrl+C to stop early.\n")

try:
    sniff(prn=cb, store=False, timeout=15)
except KeyboardInterrupt:
    pass

print("\n── IP Protocol breakdown ─────────────────────────────")
for k, v in proto_counter.most_common():
    print(f"  {k:<15} {v:>6} packets")

print("\n── Top 20 ports seen ─────────────────────────────────")
for k, v in port_counter.most_common(20):
    print(f"  {k:<20} {v:>6}")

print("\n── Service labels (what chart will show) ─────────────")
for k, v in service_counter.most_common():
    bar = "█" * min(v // 2, 40)
    print(f"  {k:<15} {v:>5}  {bar}")

print("\nIf everything shows as TCP here, your network traffic is")
print("genuinely all TCP and the chart is correct.")
print("Try opening http:// sites (not https), running a ping,")
print("or doing a DNS lookup to generate variety.")