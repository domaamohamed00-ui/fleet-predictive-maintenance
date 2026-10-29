import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_auc_score

# 1. تحميل النموذج الفائز والأعمدة المعيارية
print("1/4 جاري تحميل النموذج وقواعد البيانات...")
model_obj = joblib.load("maintenance_model.pkl")
features = joblib.load("model_features.pkl")

if isinstance(model_obj, tuple):
    clf, scaler = model_obj
else:
    clf, scaler = model_obj, None

stream_file = Path("maint_stream_out/maintenance_events.ndjson")
truth_file = Path("maint_stream_out/maintenance_truth.csv")

# 2. قراءة وتنظيف أحداث الـ Stream
if not stream_file.exists():
    print(f"في انتظار إرسال بيانات البث... قم بتشغيل fleet_maintenance_stream.py")
    while not stream_file.exists():
        time.sleep(1)

print("2/4 جاري قراءة وتنظيف بيانات البث المباشر...")
valid_records = []
seen_keys = set()
faults = {"malformed": 0, "out_of_range": 0, "duplicate": 0}

with open(stream_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        # أ) فحص سلامة الـ JSON
        try:
            ev = json.loads(line)
        except Exception:
            faults["malformed"] += 1
            continue
        
        # ب) التحقق من وجود المفاتيح الأساسية
        if "vehicle_id" not in ev or "snapshot_date" not in ev:
            faults["malformed"] += 1
            continue
        
        # ج) استبعاد القيم غير المنطقية
        if ev.get("odometer_km", 0) < 0:
            faults["out_of_range"] += 1
            continue
        
        # د) إزالة التكرار
        key = (ev["vehicle_id"], ev["snapshot_date"])
        if key in seen_keys:
            faults["duplicate"] += 1
            continue
        seen_keys.add(key)
        
        valid_records.append(ev)

print(f"تمت تصفية البيانات: {len(valid_records):,} حدث صالح | الأخطاء المستبعدة: {faults}")

# 3. تطبيق النموذج والتنبؤ بالخطر
print("3/4 جاري إجراء التنبؤات اللحظية بواسطة LightGBM...")
df_stream = pd.DataFrame(valid_records)

cat_cols = [
    'vehicle_type', 'vehicle_model', 'fuel_type', 'transmission_type', 
    'tire_condition', 'brake_condition', 'battery_status'
]
df_encoded = pd.get_dummies(df_stream, columns=cat_cols, drop_first=False)
X_stream = df_encoded.reindex(columns=features, fill_value=0)

if scaler:
    X_stream_scaled = scaler.transform(X_stream)
    y_prob = clf.predict_proba(X_stream_scaled)[:, 1]
else:
    y_prob = clf.predict_proba(X_stream)[:, 1]

df_stream["predicted_risk"] = np.round(y_prob, 4)
df_stream["predicted_need"] = (y_prob >= 0.5).astype(int)

# 4. التقييم النهائي بالدمج مع ملف الحقيقة (Ground Truth)
print("\n4/4 تقييم التنبؤات المباشرة مقارنة بالحقيقة...")
if truth_file.exists():
    df_truth = pd.read_csv(truth_file)
    merged = pd.merge(df_stream, df_truth, on=["vehicle_id", "snapshot_date"], how="inner")
    
    print("\n=======================================================")
    print("          تقرير أداء النموذج على الـ Stream            ")
    print("=======================================================")
    print(classification_report(merged["Need_Maintenance"], merged["predicted_need"], zero_division=0))
    print(f"ROC-AUC Score: {roc_auc_score(merged['Need_Maintenance'], merged['predicted_risk']):.4f}")
    
    print("\nعينة للشاحنات عالية المخاطر المطلوبة للصيانة الفورية:")
    print(merged[merged["predicted_need"] == 1][["vehicle_id", "snapshot_date", "predicted_risk", "Need_Maintenance"]].head(10).to_string(index=False))
else:
    print("ملف الحقيقة غير موجود حالياً للقياس.")