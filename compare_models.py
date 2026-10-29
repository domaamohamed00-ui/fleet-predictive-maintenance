import pandas as pd
import numpy as np
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import xgboost as xgb
import lightgbm as lgb

# 1. تحميل البيانات التاريخية
print("1/5 جاري تحميل البيانات التاريخية (450,000 صف)...")
df = pd.read_csv("maint_out/maintenance_all.csv")

# 2. معالجة الأعمدة وتجهيز البيانات
print("2/5 جاري معالجة الأعمدة وتنسيق البيانات...")
cat_cols = [
    'vehicle_type', 'vehicle_model', 'fuel_type', 'transmission_type', 
    'tire_condition', 'brake_condition', 'battery_status'
]
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)

ignore_cols = ['snapshot_date', 'vehicle_id', 'last_service_date', 'Need_Maintenance']
features = [c for c in df_encoded.columns if c not in ignore_cols]

X = df_encoded[features].fillna(0)
y = df['Need_Maintenance']

# 3. تقسيم البيانات (80% تدريب - 20% اختبار)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# تقييس البيانات للنماذج التي تحتاج Scaling (Logistic & SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. تعريف النماذج الـ 5 المحددة مع ضبط البارامترات ضد Overfitting
models = {
    "Logistic Regression": (
        LogisticRegression(C=0.1, max_iter=1000, random_state=42), 
        True
    ),
    "SVM (LinearSVC Calibrated)": (
        CalibratedClassifierCV(LinearSVC(C=0.1, random_state=42, dual=False, max_iter=2000)), 
        True
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, max_features='sqrt', random_state=42, n_jobs=-1), 
        False
    ),
    "XGBoost": (
        xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1), 
        False
    ),
    "LightGBM": (
        lgb.LGBMClassifier(n_estimators=100, max_depth=5, num_leaves=31, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1), 
        False
    )
}

# 5. تدريب وتقييم كل نموذج والتحقق من Overfitting
print("\n3/5 بدء تدريب النماذج الـ 5 ومقارنة الأداء في الـ Train والـ Test...\n")
results = []
best_auc = 0.0
best_model_name = ""
best_model_obj = None

for name, (model, use_scaler) in models.items():
    t0 = time.time()
    xtr = X_train_scaled if use_scaler else X_train
    xte = X_test_scaled if use_scaler else X_test
    
    # التدريب
    model.fit(xtr, y_train)
    train_time = time.time() - t0
    
    # حساب دقة التدريب للتحقق من عدم وجود Overfitting
    y_train_pred = model.predict(xtr)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    # حساب دقة الاختبار والقياسات الأخرى
    y_pred = model.predict(xte)
    y_prob = model.predict_proba(xte)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    test_acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    
    results.append({
        "Model": name,
        "Train Acc": round(train_acc, 4),
        "Test Acc": round(test_acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc, 4),
        "Time (s)": round(train_time, 2)
    })
    
    # تحديد النموذج الأفضل بناءً على ROC-AUC
    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_model_obj = (model, scaler) if use_scaler else model

# 6. طباعة الجدول النهائي
results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
print("=========================================================================================")
print("                               جدول مقارنة أداء النماذج الـ 5                            ")
print("=========================================================================================")
print(results_df.to_string(index=False))
print("=========================================================================================")

# 7. حفظ النموذج الأفضل وقائمة الأعمدة
print(f"\n4/5 النموذج الأفضل هو: [{best_model_name}] بمقياس ROC-AUC قدره {best_auc:.4f}")
joblib.dump(best_model_obj, "maintenance_model.pkl")
joblib.dump(features, "model_features.pkl")
print("5/5 تم حفظ النموذج الفائز في 'maintenance_model.pkl' وأعمدة البيانات في 'model_features.pkl' بنجاح!")