import pickle
import pandas as pd
from datetime import datetime

# =========================
# Load model & metadata
# =========================

with open("./SOCTest/isolation_audit_model.pkl", "rb") as f:
    model = pickle.load(f)

FEATURE_COLS = [
    "event_count",
    "failed_ratio",
    "privileged_ratio",
    "unique_event_types"
]

# (ถ้าใช้ scaler)
# with open("./SOCTest/scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# =========================
# Prepare new data
# =========================

df_events_new = normalize_events(events_new)
df_sessions_new = build_session_features(df_events_new)

# Guard: ไม่มี session
if df_sessions_new.empty:
    print("❌ No session data to analyze")
    exit()

# Guard: session น้อยเกินไป
if len(df_sessions_new) < 3:
    df_sessions_new["anomaly"] = 1
    df_sessions_new["score"] = 0.0
    print("⚠️ Not enough sessions for anomaly detection")
    print(df_sessions_new)
    exit()

# =========================
# Feature alignment
# =========================

X_new = df_sessions_new.reindex(
    columns=FEATURE_COLS,
    fill_value=0
)

# ถ้าใช้ scaler
# X_new = scaler.transform(X_new)

# =========================
# Anomaly detection
# =========================

df_sessions_new["anomaly"] = model.predict(X_new)          # -1 = anomaly
df_sessions_new["score"] = model.decision_function(X_new) # ยิ่งต่ำยิ่งแปลก

# =========================
# Severity level (SOC-friendly)
# =========================

df_sessions_new["severity"] = pd.cut(
    df_sessions_new["score"],
    bins=[-1, -0.2, 0, 1],
    labels=["high", "medium", "low"]
)

# =========================
# Model metadata
# =========================

df_sessions_new["model_version"] = "isolation_forest_v1"
df_sessions_new["checked_at"] = datetime.utcnow()

print("\n=== Session Anomaly Result ===")
print(df_sessions_new)

# =========================
# Map anomaly → raw events
# =========================

anomalous_sessions = df_sessions_new[
    df_sessions_new["anomaly"] == -1
]["session"]

suspicious_events = df_events_new[
    df_events_new["session"].isin(anomalous_sessions)
]

# =========================
# Human-readable alert
# =========================

for _, row in df_sessions_new[df_sessions_new["anomaly"] == -1].iterrows():
    session_id = row["session"]
    score = row["score"]
    severity = row["severity"]

    print(f"\n🚨 ALERT | session={session_id} | score={score:.3f} | severity={severity}")
    print(suspicious_events[suspicious_events["session"] == session_id].head())

