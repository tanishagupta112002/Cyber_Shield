import pandas as pd
import joblib
import hashlib
from log_tamper_detector import predict_logs_dataframe

# --- Step 1: Load original hashes to see what model expects ---
original_hashes = joblib.load("original_hashes.joblib")
print("[i] Original hashes loaded:")
for fname, h in original_hashes.items():
    print(fname, h)

# --- Step 2: Create a test DataFrame with one of the files ---
test_file = list(original_hashes.keys())[0]  # pick first file
correct_hash = original_hashes[test_file]

df_test = pd.DataFrame([{
    "Filename": test_file,
    "Event_ID": 123,
    "Level": 4,
    "Service": "Edge",
    "Timestamp": "2025-10-11 20:00:00",
    "TimeGenerated": "2025-10-11 20:05:00",
    "Hash": correct_hash,  # start with correct hash
    "Class": "Normal"
}])

# --- Step 3: Run prediction with correct hash (should be Normal) ---
res_normal = predict_logs_dataframe(df_test)
print("\n[i] With correct hash:")
print(res_normal)

# --- Step 4: Tamper the hash manually ---
df_test.loc[0, 'Hash'] = "tampered_hash_12345"

# --- Step 5: Run prediction with tampered hash (should detect Tampered) ---
res_tampered = predict_logs_dataframe(df_test)
print("\n[i] After tampering hash:")
print(res_tampered)
