# save as: log_tamper_detector.py
import os
import time
import hashlib
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle

# ----------------------------
# Config
# ----------------------------
LOG_FOLDER = "logs"                  # monitored folder (create if not exists)
MODEL_PATH = "model.joblib"
OHE_PATH = "ohe.joblib"
LE_PATH = "le.joblib"
ORIG_HASHES_PATH = "original_hashes.joblib"
CONFIDENCE_THRESHOLD = 0.60         # if model confidence < this -> mark Suspicious
RANDOM_STATE = 42

# ----------------------------
# Utility: compute sha256 for a string content
# ----------------------------
def compute_hash_from_string(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()

# ----------------------------
# Load / prepare initial labeled dataset
# ----------------------------
def load_initial_dataset(csv_path="../dataset/initial_labeled.csv"):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"[i] Loaded initial labeled dataset from {csv_path}, {len(df)} rows.")
    else:
        # fallback tiny example dataset
        print("[i] initial_labeled.csv not found — using built-in tiny example dataset.")
        data = [
            ["Edge_1", 10001, 4, "Edge", "2025-09-28 07:34:28", "2025-09-28 13:03:04", compute_hash_from_string("edge1_v1"), "Normal"],
            ["TPM_1", 17, 4, "TPM", "2025-09-28 07:34:28", "2025-09-28 13:03:48", compute_hash_from_string("tpm1_v1"), "Normal"],
            ["Net_1", 1073748845, 4, "Netwtw14", "2025-09-28 07:34:28", "2025-09-28 13:04:08", compute_hash_from_string("net1_v1_mod"), "Suspicious"],
            ["HP_1", 0, 4, "HP Comm Recovery", "2025-09-28 07:34:28", "2025-09-28 13:01:37", compute_hash_from_string("hp1_mod"), "Tampered"]
        ]
        df = pd.DataFrame(data, columns=["Filename","Event_ID","Level","Service","Timestamp","TimeGenerated","Hash","Class"])
    required = {"Filename","Event_ID","Level","Service","Timestamp","TimeGenerated","Hash","Class"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Initial dataset must include columns: {required}")
    return df

# ----------------------------
# Feature engineering
# ----------------------------
def featurize(df, ohe=None, fit_ohe=False):
    d = df.copy()
    d['Timestamp'] = pd.to_datetime(d['Timestamp'])
    d['TimeGenerated'] = pd.to_datetime(d['TimeGenerated'])
    d['Hour'] = d['TimeGenerated'].dt.hour
    d['DayOfWeek'] = d['TimeGenerated'].dt.dayofweek
    d['Delay_sec'] = (d['TimeGenerated'] - d['Timestamp']).dt.total_seconds().fillna(0.0)
    d['Event_ID'] = pd.to_numeric(d['Event_ID'], errors='coerce').fillna(0).astype(int)
    d['Level'] = pd.to_numeric(d['Level'], errors='coerce').fillna(0).astype(int)

    # One-hot encode Service
    if ohe is None and fit_ohe:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        svc_enc = ohe.fit_transform(d[['Service']])
    elif ohe is not None and fit_ohe:
        svc_enc = ohe.fit_transform(d[['Service']])
    elif ohe is not None and not fit_ohe:
        svc_enc = ohe.transform(d[['Service']])
    else:
        svc_enc = np.zeros((len(d),1))

    svc_cols = list(ohe.get_feature_names_out(['Service'])) if ohe else []
    svc_df = pd.DataFrame(svc_enc, columns=svc_cols, index=d.index)
    feat = pd.concat([d[['Event_ID','Level','Hour','DayOfWeek','Delay_sec']], svc_df], axis=1).fillna(0)
    return feat, ohe

# ----------------------------
# Build and persist model + encoders
# ----------------------------
def init_or_load_model(initial_df):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    feat_init, ohe = featurize(initial_df, ohe=ohe, fit_ohe=True)
    le = LabelEncoder()
    labels = le.fit_transform(initial_df['Class'])
    X, y = shuffle(feat_init, labels, random_state=RANDOM_STATE)
    clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=RANDOM_STATE)
    clf = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)   # SVM version

    clf.partial_fit(X, y, classes=np.unique(y))
    # persist
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(ohe, OHE_PATH)
    joblib.dump(le, LE_PATH)
    original_hashes = {row['Filename']: row['Hash'] for _, row in initial_df.iterrows()}
    joblib.dump(original_hashes, ORIG_HASHES_PATH)
    print(f"[i] Model and encoders saved. Model trained on {len(initial_df)} initial rows.")
    return clf, ohe, le, original_hashes

# ----------------------------
# Load persisted model (if exists) else init
# ----------------------------
def load_or_init():
    if os.path.exists(MODEL_PATH) and os.path.exists(OHE_PATH) and os.path.exists(LE_PATH) and os.path.exists(ORIG_HASHES_PATH):
        clf = joblib.load(MODEL_PATH)
        ohe = joblib.load(OHE_PATH)
        le = joblib.load(LE_PATH)
        original_hashes = joblib.load(ORIG_HASHES_PATH)
        print("[i] Loaded persisted model, encoders and original hashes.")
    else:
        init_df = load_initial_dataset()
        clf, ohe, le, original_hashes = init_or_load_model(init_df)
    return clf, ohe, le, original_hashes

# ----------------------------
# Decide final class using hash-check + model confidence
# ----------------------------
# ----------------------------
# Decide final class using hash-check + model confidence
# ----------------------------
def decide_class(pred_probs, pred_label_idx, le, filename, current_hash, original_hashes):
    """
    Determines final log class:
    1️⃣ Hash mismatch → Tampered (guaranteed)
    2️⃣ AI low confidence → Suspicious
    3️⃣ Otherwise → AI predicted class (Normal/Suspicious)
    """
    # 1️⃣ HASH MISMATCH FIRST
    if filename in original_hashes and current_hash != original_hashes[filename]:
        return "Tampered", 1.0  # confidence 1.0 for clarity

    # 2️⃣ AI prediction next
    pred_prob = pred_probs[pred_label_idx]
    pred_class = le.inverse_transform([pred_label_idx])[0]

    # 3️⃣ Low confidence mark Suspicious
    if pred_prob < CONFIDENCE_THRESHOLD and pred_class != "Tampered":
        return "Suspicious", pred_prob

    return pred_class, pred_prob

# ----------------------------
# Process new logs
# ----------------------------
def process_new_logs_df(df_new, clf, ohe, le, original_hashes, persist_after_update=True):
    if df_new.empty:
        return []
    feat, _ = featurize(df_new, ohe=ohe, fit_ohe=False)
    pred_probs = clf.predict_proba(feat)
    pred_idxs = np.argmax(pred_probs, axis=1)
    results = []
    for i, row in df_new.reset_index(drop=True).iterrows():
        fname = str(row['Filename'])
        cur_hash = str(row['Hash']) if 'Hash' in row and pd.notna(row['Hash']) else ""
        pred_idx = int(pred_idxs[i])
        final_class, confidence = decide_class(pred_probs[i], pred_idx, le, fname, cur_hash, original_hashes)
        results.append({
            "Filename": fname,
            "Predicted": final_class,
            "Confidence": float(confidence),
            "Model_pred": le.inverse_transform([pred_idx])[0],
            "Model_confidence": float(np.max(pred_probs[i]))
        })
    for r in results:
        print(f"[PRED] {r['Filename']} -> {r['Predicted']} (model:{r['Model_pred']} conf:{r['Model_confidence']:.2f}) final_conf:{r['Confidence']:.2f}")
    if 'Class' in df_new.columns:
        try:
            y_new = le.transform(df_new['Class'])
        except Exception:
            print("[w] Verified labels contain unknown class(es). Skipping online update.")
            return results
        clf.partial_fit(feat, y_new)
        print(f"[i] Model updated with {len(df_new)} verified rows.")
        if persist_after_update:
            joblib.dump(clf, MODEL_PATH)
            for _, row in df_new.iterrows():
                if 'Hash' in row and pd.notna(row['Hash']):
                    original_hashes[row['Filename']] = row['Hash']
            joblib.dump(original_hashes, ORIG_HASHES_PATH)
            print("[i] Persisted updated model and original_hashes.")
    return results

# ----------------------------
# Watchdog handler
# ----------------------------
class CSVLogHandler(FileSystemEventHandler):
    def __init__(self, clf, ohe, le, orig_hashes):
        super().__init__()
        self.clf = clf
        self.ohe = ohe
        self.le = le
        self.orig_hashes = orig_hashes

    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith(".csv"):
            return
        print(f"[i] New file detected: {event.src_path}")
        try:
            df_new = pd.read_csv(event.src_path)
        except Exception as e:
            print(f"[e] Failed to read CSV {event.src_path}: {e}")
            return
        required = {"Filename","Event_ID","Level","Service","Timestamp","TimeGenerated","Hash"}
        if not required.issubset(df_new.columns):
            print(f"[w] CSV missing required cols: {required}. Skipping.")
            return
        process_new_logs_df(df_new, self.clf, self.ohe, self.le, self.orig_hashes)

# ----------------------------
# Backend API integration
# ----------------------------
def predict_logs_dataframe(df_new):
    """
    Given a pandas DataFrame of logs, run AI model and return predictions list (dicts).
    This is used by backend Flask API before sending logs to frontend.
    """
    clf, ohe, le, original_hashes = load_or_init()
    results = process_new_logs_df(
        df_new,
        clf,
        ohe,
        le,
        original_hashes,
        persist_after_update=False  # don't retrain model on every request
    )
    return results

# ----------------------------
# HASHCHAIN INTEGRATION
# ----------------------------
from extensions import db
from models import LogEvent, LogFile

def verify_hashchain(system_id=None):
    """
    Walk logs in seq_no order per system (or all) and detect:
    - missing seq_no (deleted logs)
    - hash mismatch (tampered logs)
    Returns list of dicts with results.
    """
    query = LogEvent.query.join(LogFile)
    if system_id:
        query = query.filter(LogFile.system_id == system_id)
    logs = query.order_by(LogEvent.seq_no).all()

    if not logs:
        print("[i] No logs found for hashchain verification.")
        return []

    results = []
    last_seq = 0

    for log in logs:
        tampered_flag = False
        reason = ""

        # Check seq_no continuity
        if last_seq != 0 and log.seq_no != last_seq + 1:
            tampered_flag = True
            reason += f"Seq_no gap: expected {last_seq+1}, got {log.seq_no}. "

        # Compute canonical hash like agent/backend
        payload = f"{log.seq_no}||{log.time_generated}||{log.eventid}||{log.level}||{log.source_name}||{log.message or ''}"
        recomputed = hashlib.sha256(((log.prev_hash or '') + payload).encode('utf-8')).hexdigest()

        if recomputed != (log.hash or "").strip().lower():
            tampered_flag = True
            reason += "Hash mismatch. "

        results.append({
            "Filename": log.log_file.filename,
            "Seq_No": log.seq_no,
            "Hash": log.hash,
            "Prev_Hash": log.prev_hash,
            "Tampered": tampered_flag,
            "Reason": reason.strip()
        })

        last_seq = log.seq_no

    tampered_count = sum(1 for r in results if r["Tampered"])
    print(f"[i] Hashchain verification done. {tampered_count}/{len(results)} logs flagged as tampered.")
    return results

# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(LOG_FOLDER, exist_ok=True)
    clf, ohe, le, original_hashes = load_or_init()
    event_handler = CSVLogHandler(clf, ohe, le, original_hashes)
    observer = Observer()
    observer.schedule(event_handler, path=LOG_FOLDER, recursive=False)
    observer.start()
    print(f"[i] Monitoring folder '{LOG_FOLDER}' for new CSV logs. Drop CSVs there to trigger detection.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[i] Stopping observer...")
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()

import pandas as pd
import hashlib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

MODEL_FILE = os.path.join(os.path.dirname(__file__), "model.pkl")
ENC_FILE = os.path.join(os.path.dirname(__file__), "encoders.pkl")

# --- Hash check function ---
def verify_hash(row):
    calculated = hashlib.sha256(
        f"{row['Filename']}{row['Event_ID']}{row['Level']}{row['Service']}{row['Timestamp']}{row['TimeGenerated']}".encode()
    ).hexdigest()
    return 'Tampered' if calculated != row['Hash'] else 'Normal'

# --- Train / Persist model ---
def train_model(csv_file="logs.csv"):
    df = pd.read_csv(csv_file)
    df['Class'] = df.apply(verify_hash, axis=1)

    features = ['Filename', 'Event_ID', 'Level', 'Service']
    X = df[features].copy()
    y = df['Class']

    encoders = {}
    for col in ['Filename', 'Service', 'Level']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # Persist model and encoders
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)
    with open(ENC_FILE, "wb") as f:
        pickle.dump(encoders, f)
    print("[i] Model and encoders saved.")

# --- Load model & encoders ---
def load_model():
    with open(MODEL_FILE, "rb") as f:
        clf = pickle.load(f)
    with open(ENC_FILE, "rb") as f:
        encoders = pickle.load(f)
    return clf, encoders

# --- Predict DataFrame ---
def predict_logs_dataframe(df):
    clf, encoders = load_model()
    df['Predicted'] = 'Normal'

    # Encode categorical columns
    for col in ['Filename', 'Service', 'Level']:
        if col in df.columns:
            le = encoders[col]
            # Handle unseen values
            df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Predict
    X = df[['Filename', 'Event_ID', 'Level', 'Service']]
    df['Predicted'] = clf.predict(X)
    return df.to_dict(orient="records")

# # log_tamper_detector.py
# import os
# import time
# import hashlib
# import joblib
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from sklearn.linear_model import SGDClassifier
# from sklearn.utils import shuffle

# # ----------------------------
# # Config
# # ----------------------------
# LOG_FOLDER = "logs"
# MODEL_PATH = "model.joblib"
# OHE_PATH = "ohe.joblib"
# LE_PATH = "le.joblib"
# ORIG_HASHES_PATH = "original_hashes.joblib"
# CONFIDENCE_THRESHOLD = 0.60
# RANDOM_STATE = 42

# # ----------------------------
# # Utility
# # ----------------------------
# def compute_hash_from_string(s: str) -> str:
#     h = hashlib.sha256()
#     h.update(s.encode("utf-8"))
#     return h.hexdigest()

# # ----------------------------
# # Feature engineering
# # ----------------------------
# def featurize(df, ohe=None, fit_ohe=False):
#     d = df.copy()
#     d['Timestamp'] = pd.to_datetime(d['Timestamp'])
#     d['TimeGenerated'] = pd.to_datetime(d['TimeGenerated'])
#     d['Hour'] = d['TimeGenerated'].dt.hour
#     d['DayOfWeek'] = d['TimeGenerated'].dt.dayofweek
#     d['Delay_sec'] = (d['TimeGenerated'] - d['Timestamp']).dt.total_seconds().fillna(0.0)
#     d['Event_ID'] = pd.to_numeric(d['Event_ID'], errors='coerce').fillna(0).astype(int)
#     d['Level'] = pd.to_numeric(d['Level'], errors='coerce').fillna(0).astype(int)

#     if ohe is None and fit_ohe:
#         ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#         svc_enc = ohe.fit_transform(d[['Service']])
#     elif ohe is not None and fit_ohe:
#         svc_enc = ohe.fit_transform(d[['Service']])
#     elif ohe is not None and not fit_ohe:
#         svc_enc = ohe.transform(d[['Service']])
#     else:
#         svc_enc = np.zeros((len(d),1))

#     svc_cols = list(ohe.get_feature_names_out(['Service'])) if ohe else []
#     svc_df = pd.DataFrame(svc_enc, columns=svc_cols, index=d.index)
#     feat = pd.concat([d[['Event_ID','Level','Hour','DayOfWeek','Delay_sec']], svc_df], axis=1).fillna(0)
#     return feat, ohe

# # ----------------------------
# # Train model function
# # ----------------------------
# def train_model(csv_path):
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(f"CSV not found: {csv_path}")

#     df = pd.read_csv(csv_path)
#     required = {"Filename","Event_ID","Level","Service","Timestamp","TimeGenerated","Hash","Class"}
#     if not required.issubset(df.columns):
#         raise ValueError(f"CSV must include columns: {required}")

#     # One-hot + labels
#     ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#     feat, ohe = featurize(df, ohe=ohe, fit_ohe=True)
#     le = LabelEncoder()
#     labels = le.fit_transform(df['Class'])
#     X, y = shuffle(feat, labels, random_state=RANDOM_STATE)

#     # Model
#     clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=RANDOM_STATE)
#     clf.partial_fit(X, y, classes=np.unique(y))

#     # Persist model, encoders, original hashes
#     joblib.dump(clf, MODEL_PATH)
#     joblib.dump(ohe, OHE_PATH)
#     joblib.dump(le, LE_PATH)
#     original_hashes = {row['Filename']: row['Hash'] for _, row in df.iterrows()}
#     joblib.dump(original_hashes, ORIG_HASHES_PATH)

#     print(f"[i] Model and encoders saved. Trained on {len(df)} rows.")

# # ----------------------------
# # Predict logs from DataFrame
# # ----------------------------
# def predict_logs_dataframe(df_new):
#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError("Model not found. Train first using train_model(csv_path)")

#     clf = joblib.load(MODEL_PATH)
#     ohe = joblib.load(OHE_PATH)
#     le = joblib.load(LE_PATH)
#     original_hashes = joblib.load(ORIG_HASHES_PATH)

#     if df_new.empty:
#         return []

#     feat, _ = featurize(df_new, ohe=ohe, fit_ohe=False)
#     pred_probs = clf.predict_proba(feat)
#     pred_idxs = np.argmax(pred_probs, axis=1)

#     results = []
#     for i, row in df_new.reset_index(drop=True).iterrows():
#         fname = str(row['Filename'])
#         cur_hash = str(row['Hash']) if 'Hash' in row and pd.notna(row['Hash']) else ""
#         pred_idx = int(pred_idxs[i])
#         pred_class = le.inverse_transform([pred_idx])[0]
#         confidence = float(pred_probs[i][pred_idx])

#         # Hash check
#         if fname in original_hashes and cur_hash != original_hashes[fname]:
#             final_class = "Tampered"
#             confidence = 1.0
#         elif confidence < CONFIDENCE_THRESHOLD and pred_class != "Tampered":
#             final_class = "Suspicious"
#         else:
#             final_class = pred_class

#         results.append({
#             "Filename": fname,
#             "Predicted": final_class,
#             "Confidence": confidence,
#             "Model_pred": pred_class,
#             "Model_confidence": float(np.max(pred_probs[i]))
#         })

#     return results
