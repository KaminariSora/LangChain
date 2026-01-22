import pandas as pd
from sklearn.ensemble import IsolationForest
from detect import results

extract_logs = results

df = pd.DataFrame(extract_logs)

df['exe_id'] = pd.factorize(df['exe'])[0]

clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(df[['uid', 'exe_id', 'is_privileged']])

# อันนี้แค่ทดสอบดูว่าโมเดลเห็นข้อมูลที่เรายัดเข้าไปเป็นยังไง
df['anomaly_score'] = clf.predict(df[['uid', 'exe_id', 'is_privileged']])

print(df['anomaly_score'])

# หลังจากนี้ก็แค่จับ fit train model เป็น .pkl อย่างเดียว