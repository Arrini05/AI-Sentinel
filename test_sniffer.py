# ===== test_sniffer.py =====

"""
Simple scapy test to verify packet capture works.
Run: python test_sniffer.py
"""

from scapy.all import sniff

def packet_callback(packet):
    """Print packet summary."""
    print(packet.summary())


if __name__ == "__main__":
    print("Capturing 10 packets... (Ctrl+C to stop)")
    try:
        sniff(prn=packet_callback, store=False, count=10)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Error: {e}")