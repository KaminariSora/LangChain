import re

with open('./SOCTest/data/test.txt', 'r') as file:
    log_data = file.read()

def normalize_audit_log(data):
    lines = data.strip().split('\n')
    events = []

    for line in lines:
        pairs = re.findall(r'(\w+)=(?:\"([^\"]*)\"|\'([^\']*)\'|(\S+))', line)
        raw = {k: (v1 or v2 or v3) for k, v1, v2, v3 in pairs}

        exe = raw.get("exe", "")
        acct = raw.get("acct", "")
        uid = raw.get("uid", "")
        res = raw.get("res", "")

        # --- Normalize fields ---
        event = {
            "event_type": raw.get("type"),
            "user": acct if acct else uid,
            "target_user": "root" if acct == "root" or uid == "0" else None,
            "exe": exe,
            "command": raw.get("cmd"),
            "session": raw.get("ses"),
            "terminal": raw.get("terminal"),
            "result": "success" if res == "success" else "failed",
        }


        events.append(event)

    return events


# --- การแสดงผล ---
events = normalize_audit_log(log_data)

for e in events:
    print(e)

# print("event lenght: ",len(events))
# print(events[0])

# import json

# file_name = "./SOCTest/audit_events.json"

# with open(file_name, "w", encoding="utf-8") as f:
#     json.dump(events, f, indent=4, ensure_ascii=False)