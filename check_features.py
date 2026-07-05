"""
check_features.py — Feature rankings for ALL datasets in one script.
Run from project root:  .venv\Scripts\python check_features_all.py

Automatically detects which selectors exist and runs only those.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import joblib
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import config

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE NAME MAPS
# ══════════════════════════════════════════════════════════════════════════════

def get_nsl_names(n_total):
    numeric = [
        "duration","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
        "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
        "num_root","num_file_creations","num_shells","num_access_files",
        "num_outbound_cmds","is_host_login","is_guest_login","count","srv_count",
        "serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
        "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
        "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
        "dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    ]
    protocol_type = ["icmp","tcp","udp"]
    service = [
        "IRC","X11","Z39_50","aol","auth","bgp","courier","csnet_ns","ctf",
        "daytime","discard","domain","domain_u","echo","eco_i","ecr_i","efs",
        "exec","finger","ftp","ftp_data","gopher","harvest","hostnames","http",
        "http_2784","http_443","http_8001","imap4","iso_tsap","klogin","kshell",
        "ldap","link","login","mtp","name","netbios_dgm","netbios_ns","netbios_ssn",
        "netstat","nnsp","nntp","ntp_u","other","pm_dump","pop_2","pop_3","printer",
        "private","red_i","remote_job","rje","shell","smtp","sql_net","ssh",
        "sunrpc","supdup","systat","telnet","tim_i","time","urh_i","urp_i","uucp",
        "uucp_path","vmnet","whois",
    ]
    flag = ["OTH","REJ","RSTO","RSTOS0","RSTR","S0","S1","S2","S3","SF","SH"]
    n_num  = len(numeric)
    n_ohe  = n_total - n_num
    ohe = (
        [f"protocol_type={c}" for c in protocol_type] +
        [f"service={c}"        for c in service[:20]]  +
        [f"flag={c}"           for c in flag[:11]]
    )
    return numeric + ohe[:n_ohe]


def get_unsw_names(n_total):
    numeric = [
        "dur","spkts","dpkts","sbytes","dbytes","rate","sttl","dttl",
        "sload","dload","sloss","dloss","sinpkt","dinpkt","sjit","djit",
        "swin","stcpb","dtcpb","dwin","tcprtt","synack","ackdat",
        "smean","dmean","trans_depth","response_body_len","ct_srv_src",
        "ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm","ct_dst_sport_ltm",
        "ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd","ct_flw_http_mthd",
        "ct_src_ltm","ct_srv_dst","is_sm_ips_ports",
    ]
    proto   = ["tcp","udp","unas","ospf","arp","sctp","gre","icmp",
               "rtp","pim","igmp","ip","ipv6-icmp","swipe","pri",
               "vrrp","cbt","skip","zero","other_proto"]
    service = ["-","dns","http","smtp","ftp-data","ftp","ssh","pop3",
               "snmp","ssl","irc","radius","dhcp","other_svc"]
    state   = ["FIN","INT","CON","REQ","RST","CLO","PAR","ECO",
               "URN","no","ACC","other_st"]
    n_num   = len(numeric)
    n_ohe   = n_total - n_num
    ohe = (
        [f"proto={p}"   for p in proto]   +
        [f"service={s}" for s in service] +
        [f"state={s}"   for s in state]
    )
    return numeric + ohe[:n_ohe]


def get_live_names(n_total):
    """Live dataset: read column names directly from the CSV."""
    import pandas as pd
    live_path = config.LIVE_DATASET
    if os.path.isfile(live_path):
        df = pd.read_csv(live_path, nrows=1)
        cols = [c for c in df.columns if c != "label"]
        return cols[:n_total]
    # Fallback: known 6-feature Scapy schema
    return ["packet_length","src_port","dst_port","protocol","ttl","tcp_flags"]


# ══════════════════════════════════════════════════════════════════════════════
# GROUP LABELS  (works across all three datasets)
# ══════════════════════════════════════════════════════════════════════════════

GROUPS = {
    # NSL-KDD
    "src_bytes":"Traffic Volume",   "dst_bytes":"Traffic Volume",
    "diff_srv_rate":"Conn Rate",    "same_srv_rate":"Conn Rate",
    "count":"Conn Rate",            "srv_count":"Conn Rate",
    "srv_diff_host_rate":"Conn Rate",
    "dst_host_srv_count":"Host",    "dst_host_same_srv_rate":"Host",
    "dst_host_diff_srv_rate":"Host","dst_host_count":"Host",
    "dst_host_same_src_port_rate":"Host","dst_host_srv_diff_host_rate":"Host",
    "dst_host_serror_rate":"Error", "dst_host_srv_serror_rate":"Error",
    "dst_host_rerror_rate":"Error", "dst_host_srv_rerror_rate":"Error",
    "serror_rate":"Error",          "srv_serror_rate":"Error",
    "rerror_rate":"Error",          "srv_rerror_rate":"Error",
    "logged_in":"Auth",             "duration":"Basic",
    # UNSW-NB15
    "sbytes":"Traffic Volume",      "dbytes":"Traffic Volume",
    "spkts":"Traffic Volume",       "dpkts":"Traffic Volume",
    "rate":"Traffic Volume",        "dur":"Basic",
    "sttl":"Network",               "dttl":"Network",
    "sload":"Traffic Volume",       "dload":"Traffic Volume",
    "sloss":"Error",                "dloss":"Error",
    "sinpkt":"Timing",              "dinpkt":"Timing",
    "sjit":"Timing",                "djit":"Timing",
    "swin":"Network",               "dwin":"Network",
    "tcprtt":"Timing",              "synack":"Timing",
    "ackdat":"Timing",              "smean":"Traffic Volume",
    "dmean":"Traffic Volume",       "trans_depth":"App Layer",
    "response_body_len":"App Layer",
    "ct_srv_src":"Flow Counter",    "ct_state_ttl":"Flow Counter",
    "ct_dst_ltm":"Flow Counter",    "ct_src_dport_ltm":"Flow Counter",
    "ct_dst_sport_ltm":"Flow Counter","ct_dst_src_ltm":"Flow Counter",
    "ct_ftp_cmd":"App Layer",       "ct_flw_http_mthd":"App Layer",
    "ct_src_ltm":"Flow Counter",    "ct_srv_dst":"Flow Counter",
    "is_ftp_login":"Auth",          "is_sm_ips_ports":"Network",
    # Live
    "packet_length":"Basic",        "src_port":"Network",
    "dst_port":"Network",           "protocol":"Network",
    "ttl":"Network",                "tcp_flags":"Network",
}


# ══════════════════════════════════════════════════════════════════════════════
# PRINT FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def print_ranking(dataset_label, raw_features, selector, name_fn, info_lines):
    scores  = selector.scores_
    indices = np.argsort(scores)[::-1]
    n_total = len(scores)
    names   = name_fn(n_total)
    k       = selector.k

    print("\n" + "█"*75)
    print(f"  {dataset_label}")
    print("█"*75)
    for line in info_lines:
        print(f"  {line}")
    print(f"  Total features after OHE : {n_total}")
    print(f"  Selected (SelectKBest k) : {k}")
    print("─"*75)
    print(f"  {'Rank':<5} {'MI Score':>9}  {'Feature Name':<38} Group")
    print("─"*75)

    top_names = []
    for rank, idx in enumerate(indices[:k], 1):
        name  = names[idx] if idx < len(names) else f"feature_{idx}"
        base  = name.split("=")[0] if "=" in name else name
        grp   = GROUPS.get(base, "Categorical" if "=" in name else "Other")
        print(f"  {rank:<5} {scores[idx]:>9.4f}  {name:<38} {grp}")
        top_names.append(name)

    print("\n  ── By Group ─────────────────────────────────────────────────────")
    grp_count = Counter(
        "Categorical" if "=" in n else GROUPS.get(n, "Other")
        for n in top_names
    )
    for g, c in sorted(grp_count.items(), key=lambda x: -x[1]):
        bar = "█" * c
        print(f"  {g:<20} {bar}  ({c})")


# ══════════════════════════════════════════════════════════════════════════════
# DATASET DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

DATASETS = [
    {
        "key":       "nsl_kdd",
        "label":     "NSL-KDD  (KDDTrain+.txt)",
        "raw_desc":  "42 columns → 38 numeric + 3 categorical (protocol_type, service, flag)",
        "name_fn":   get_nsl_names,
    },
    {
        "key":       "unsw",
        "label":     "UNSW-NB15  (unsw_nb15_training-set.parquet)",
        "raw_desc":  "46 features → numeric + categorical (proto, service, state)",
        "name_fn":   get_unsw_names,
    },
    {
        "key":       "live_dataset",
        "label":     "Live Dataset  (live_dataset.csv)",
        "raw_desc":  "Packet-level features extracted by Scapy (Real NIC mode)",
        "name_fn":   get_live_names,
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

found = 0
missing = []

for ds in DATASETS:
    path = os.path.join(config.MODEL_DIR, f"selector_{ds['key']}.joblib")
    if not os.path.isfile(path):
        missing.append(ds["key"])
        continue

    try:
        selector = joblib.load(path)
        print_ranking(
            dataset_label = ds["label"],
            raw_features  = ds["raw_desc"],
            selector      = selector,
            name_fn       = ds["name_fn"],
            info_lines    = [ds["raw_desc"]],
        )
        found += 1
    except Exception as e:
        print(f"\n⚠️  Failed to load {ds['key']}: {e}")

print("\n" + "="*75)
print(f"  Completed: {found} dataset(s) evaluated.")
if missing:
    print(f"\n  ⚠️  Missing selectors for: {', '.join(missing)}")
    print("  Train those datasets first:")
    for m in missing:
        print(f"    .venv\\Scripts\\python main.py --train --dataset {m}")
print("="*75 + "\n")