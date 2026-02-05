import pandas as pd
from detect import events
import pickle

EVENT_TYPE_MAP = {
    "USER_CMD": 0,
    "SYSCALL": 1,
    "LOGIN": 2,
    "SERVICE_START": 3,
    "SERVICE_STOP": 4,
    "OTHER": 5
}

RESULT_MAP = {
    "success": 1,
    "failed": 0
}

def normalize_events(events):
    rows = []

    for e in events:
        session = e.get("session")

        rows.append({
            "session": session,
            "event_type_id": EVENT_TYPE_MAP.get(
                e.get("event_type"), EVENT_TYPE_MAP["OTHER"]
            ),
            "is_privileged": int(e.get("user") == "0"),
            "result_id": RESULT_MAP.get(e.get("result", "failed"), 0)
        })

    return pd.DataFrame(rows)

def build_session_features(df):
    rows = []

    for session_id, g in df.groupby("session"):
        rows.append({
            "session": session_id,

            # ปริมาณ
            "event_count": len(g),

            # สัดส่วน
            "failed_ratio": (g["result_id"] == 0).mean(),
            "privileged_ratio": g["is_privileged"].mean(),

            # ความหลากหลาย
            "unique_event_types": g["event_type_id"].nunique()
        })

    return pd.DataFrame(rows)


from sklearn.ensemble import IsolationForest

FEATURE_COLS = [
    "event_count",
    "failed_ratio",
    "privileged_ratio",
    "unique_event_types"
]

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

# normalize
df_events = normalize_events(events)

# build behavior
df_sessions = build_session_features(df_events)

print(df_sessions)

# train
X = df_sessions[FEATURE_COLS]
model.fit(X)

# score
df_sessions["anomaly"] = model.predict(X)      # -1 = anomaly
df_sessions["score"] = model.decision_function(X)

print(df_sessions)

anomalies = df_sessions['anomaly'] == -1
print(anomalies)

with open('iso_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("save model successful")