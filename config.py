# ===== config.py (FULLY CORRECTED & ENHANCED) =====

import os

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
NSL_MODEL_DIR = os.path.join(MODEL_DIR, "nsl_kdd")
UNSW_MODEL_DIR = os.path.join(MODEL_DIR, "unsw")
LOG_DIR = os.path.join(BASE_DIR, "logs")
ALERT_DIR = os.path.join(BASE_DIR, "alerts")

# Create directories if they don't exist
for _dir in [MODEL_DIR, NSL_MODEL_DIR, UNSW_MODEL_DIR, LOG_DIR, ALERT_DIR]:
    os.makedirs(_dir, exist_ok=True)

# NSL-KDD dataset paths
NSL_KDD_TRAIN = os.path.join(DATA_DIR, "nsl-kdd", "KDDTrain+.txt")
NSL_KDD_TEST = os.path.join(DATA_DIR, "nsl-kdd", "KDDTest+.txt")

# UNSW-NB15 dataset paths (parquet)
UNSW_TRAIN = os.path.join(DATA_DIR, "UNSW-NB15", "unsw_nb15_training-set.parquet")
UNSW_TEST = os.path.join(DATA_DIR, "UNSW-NB15", "unsw_nb15_testing-set.parquet")

# Live dataset path
LIVE_DATASET = os.path.join(DATA_DIR, "live_dataset.csv")

# Model save paths
RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_{dataset}.joblib")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_{dataset}.joblib")
MLP_MODEL_PATH = os.path.join(MODEL_DIR, "mlp_{dataset}.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor_{dataset}.joblib")

# Alert logs
ALERT_LOG_PATH = os.path.join(ALERT_DIR, "alerts.log")
ALERT_JSON_PATH = os.path.join(ALERT_DIR, "alerts.json")

# ─────────────────────────────────────────────
# Dataset selection
# ─────────────────────────────────────────────
ACTIVE_DATASET = "nsl_kdd"

# ─────────────────────────────────────────────
# NSL-KDD column names (42 columns)
# ─────────────────────────────────────────────
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

NSL_KDD_CATEGORICAL = ["protocol_type", "service", "flag"]
NSL_KDD_LABEL_COL = "label"
NSL_KDD_DROP_COLS = ["difficulty"]
NSL_KDD_NORMAL_LABEL = "normal"

# ─────────────────────────────────────────────
# UNSW-NB15
# ─────────────────────────────────────────────
UNSW_LABEL_COL = "label"
UNSW_DROP_COLS = ["id", "attack_cat"]

# ─────────────────────────────────────────────
# Train/test split
# ─────────────────────────────────────────────
TEST_SIZE = 0.2

# ─────────────────────────────────────────────
# Model hyperparameters
# ─────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 30,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "class_weight": "balanced",
}

SVM_PARAMS = {
    "C": 1.0,
    "random_state": RANDOM_SEED,
    "max_iter": 5000,
}

MLP_PARAMS = {
    "hidden_layer_sizes": (256, 128, 64),
    "activation": "relu",
    "solver": "adam",
    "max_iter": 500,
    "random_state": RANDOM_SEED,
    "early_stopping": True,
    "n_iter_no_change": 20,
    "learning_rate_init": 0.001,
    "alpha": 0.0001,
    "batch_size": 256,
}

# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────
SCALER_TYPE = "RobustScaler"  # Options: StandardScaler, RobustScaler
N_FEATURES_SELECT = 20
BALANCING_METHOD = "SMOTE"  # Options: SMOTE, ADASYN, SMOTEENN, SMOTETomek

# ─────────────────────────────────────────────
# Adaptive learning
# ─────────────────────────────────────────────
ADAPTIVE_RETRAIN_THRESHOLD = 100
ADAPTIVE_LOG_PATH = os.path.join(LOG_DIR, "adaptive.log")
ADAPTIVE_SUBSAMPLE_RATIO = 0.3
ADAPTIVE_MAX_BUFFER = 1000

# ─────────────────────────────────────────────
# Cross validation
# ─────────────────────────────────────────────
CV_N_SPLITS = 5
CV_N_REPEATS = 3
CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# ─────────────────────────────────────────────
# Live traffic simulation
# ─────────────────────────────────────────────
SIMULATION_INTERVAL_SEC = 0.5
SIMULATION_BATCH_SIZE = 10
LIVE_CAPTURE_INTERFACE = None  # Set to interface name (e.g., "eth0")

# ─────────────────────────────────────────────
# Attack type classification
# ─────────────────────────────────────────────
ATTACK_TYPES = {
    0: "Normal",
    1: "DoS",
    2: "Probe",
    3: "R2L",
    4: "U2R",
}

ATTACK_COLORS = {
    "Normal": "#34d399",
    "DoS": "#f87171",
    "Probe": "#fb923c",
    "R2L": "#c084fc",
    "U2R": "#f43f5e",
    "Unknown": "#94a3b8",
}

# NSL-KDD multi-class label → integer mapping
NSL_KDD_ATTACK_FAMILIES = {
    "normal": 0,
    # DoS
    "back": 1, "land": 1, "neptune": 1,
    "pod": 1, "smurf": 1, "teardrop": 1,
    "apache2": 1, "udpstorm": 1, "processtable": 1, "worm": 1,
    # Probe
    "ipsweep": 2, "nmap": 2, "portsweep": 2,
    "satan": 2, "mscan": 2, "saint": 2,
    # R2L
    "ftp_write": 3, "guess_passwd": 3, "imap": 3,
    "multihop": 3, "phf": 3, "spy": 3,
    "warezclient": 3, "warezmaster": 3, "sendmail": 3,
    "named": 3, "snmpgetattack": 3, "snmpguess": 3,
    "xlock": 3, "xsnoop": 3, "httptunnel": 3,
    # U2R
    "buffer_overflow": 4, "loadmodule": 4, "perl": 4,
    "rootkit": 4, "ps": 4, "sqlattack": 4, "xterm": 4,
}

# ─────────────────────────────────────────────
# Smart model auto-selection thresholds
# ─────────────────────────────────────────────
SMART_MODEL_HIGH_RATE_THRESHOLD = 50
SMART_MODEL_LOW_VARIANCE_THRESHOLD = 0.05
SMART_MODEL_ENSEMBLE_ENABLED = False
SMART_MODEL_MIN_CONFIDENCE = 0.6

# ─────────────────────────────────────────────
# DDoS / flood grouping
# ─────────────────────────────────────────────
DDOS_WINDOW_SEC = 10
DDOS_PACKET_THRESH = 20
DDOS_AUTO_BLOCK = False

# ─────────────────────────────────────────────
# GeoIP (ip-api.com free tier)
# ─────────────────────────────────────────────
GEOIP_API_URL = "http://ip-api.com/json/{ip}?fields=country,regionName,city,isp,org,as,query,status"
GEOIP_TIMEOUT = 2
GEOIP_CACHE_SIZE = 512

# Known-malicious IP list
KNOWN_MALICIOUS_IPS = {
    "185.220.101.1", "45.33.32.156", "198.20.70.114",
    "89.248.167.131", "80.82.77.33", "94.102.49.190",
}

# ─────────────────────────────────────────────
# Speed optimization
# ─────────────────────────────────────────────
MAX_TRAIN_SAMPLES = 20000  # Use 20K instead of 125K

# Deep model training settings (faster, still achieves >85% accuracy)
DEEP_MODEL_EPOCHS = 8       # EarlyStopping kicks in before this anyway
DEEP_MODEL_BATCH  = 512     # larger batch = fewer steps per epoch = faster