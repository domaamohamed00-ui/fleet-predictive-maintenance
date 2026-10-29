"""
Executive AI Fleet Control Center v3 - Real-Time Predictive Maintenance
تشغيل الواجهة: streamlit run app.py
"""
import inspect
import json
import math
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# 1. إعداد الصفحة
st.set_page_config(
    page_title="Executive AI Predictive Maintenance Center",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

PLOT_W = {"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters else {"use_container_width": True}
DF_W = {"width": "stretch"} if "width" in inspect.signature(st.dataframe).parameters else {"use_container_width": True}
PLOT_CFG = {"displayModeBar": False}

# 2. تصميم CSS فاخر مريح للعين
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 50% -10%, #1d2540 0%, #0a0d14 70%);
    color: #ffffff;
}
.block-container {
    padding-top: 1.2rem;
    max-width: 1650px;
}
.header-title {
    font-size: 2.3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00e676);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.live-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 18px;
    border-radius: 30px;
    font-size: 0.82rem;
    font-weight: 800;
    background: rgba(0, 230, 118, 0.12);
    color: #00e676;
    border: 1px solid rgba(0, 230, 118, 0.4);
}
.pulse-dot {
    width: 9px;
    height: 9px;
    background-color: #00e676;
    border-radius: 50%;
    box-shadow: 0 0 10px #00e676;
}
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    transition: transform 0.3s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: rgba(0, 210, 255, 0.5);
}
.panel-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin: 0.8rem 0 0.3rem;
    color: #00d2ff;
    border-left: 4px solid #00d2ff;
    padding-left: 10px;
}
div[data-testid="stPlotlyChart"] {
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    background: rgba(255, 255, 255, 0.015);
    padding: 4px;
}
.fc {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    margin: 5px 0;
    border-radius: 10px;
    background: rgba(255,255,255,0.035);
    border-left: 4px solid #00e676;
    font-size: 0.88rem;
}
.fc b { font-size: 0.98rem; }
.fc small { opacity: 0.7; }
.fc .r { font-weight: 800; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

# 3. تحميل الموديل والبيانات
CAT_COLS = ["vehicle_type", "vehicle_model", "fuel_type", "transmission_type",
            "tire_condition", "brake_condition", "battery_status"]

@st.cache_resource
def load_assets():
    obj = joblib.load("maintenance_model.pkl")
    feats = joblib.load("model_features.pkl")
    clf = obj[0] if isinstance(obj, tuple) else obj
    scaler = obj[1] if isinstance(obj, tuple) else None
    return clf, scaler, feats

try:
    clf, scaler, feats = load_assets()
    ASSETS_ERR = None
except Exception as e:
    ASSETS_ERR = str(e)

@st.cache_resource(max_entries=2)
def load_truth(path_str, bucket):
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        t = pd.read_csv(p, usecols=lambda c: c in ("vehicle_id", "snapshot_date", "Need_Maintenance"))
        t = t.drop_duplicates(["vehicle_id", "snapshot_date"]).set_index(["vehicle_id", "snapshot_date"])
        return t["Need_Maintenance"]
    except Exception:
        return None

# 4. دالة تنسيق الرسوم البيانية
RISK_SCALE = [[0, "#00e676"], [0.5, "#ffd600"], [0.75, "#ff6d00"], [1, "#ff1744"]]

def style_fig(fig, height=350, **kw):
    layout_defaults = dict(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.01)",
        margin=dict(l=12, r=12, t=28, b=12),
        font=dict(family="Segoe UI, Tahoma, sans-serif", size=11),
        uirevision="keep",
        legend=dict(orientation="h", y=1.12, x=0)
    )
    if "margin" in kw:
        layout_defaults["margin"].update(kw.pop("margin"))
    layout_defaults.update(kw)
    fig.update_layout(**layout_defaults)
    return fig

def show_chart(fig, key):
    st.plotly_chart(fig, key=key, config=PLOT_CFG, **PLOT_W)

def title(t):
    st.markdown(f'<div class="panel-title">{t}</div>', unsafe_allow_html=True)

# 5. قراءة البث وتجهيز التنبؤات اللحظية
def process_live_stream(path_str="maint_stream_out/maintenance_events.ndjson", max_tail=15000):
    p = Path(path_str)
    if not p.exists():
        return None, 0

    valid_records = []
    seen_keys = set()

    with open(p, "rb") as f:
        lines = f.readlines()
        total_lines = len(lines)
        recent_lines = lines[-max_tail:] if total_lines > max_tail else lines

        for l in reversed(recent_lines):
            try:
                ev = json.loads(l.decode('utf-8', errors='ignore'))
            except Exception:
                continue

            if "vehicle_id" in ev and "snapshot_date" in ev and ev.get("odometer_km", 0) >= 0:
                key = (ev["vehicle_id"], ev["snapshot_date"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    valid_records.append(ev)

    if not valid_records:
        return None, total_lines

    df = pd.DataFrame(valid_records)

    if ASSETS_ERR is None:
        enc = pd.get_dummies(df, columns=[c for c in CAT_COLS if c in df.columns], drop_first=False)
        X = enc.reindex(columns=feats, fill_value=0)
        df["predicted_risk"] = np.round(clf.predict_proba(scaler.transform(X) if scaler else X)[:, 1], 4)
    else:
        df["predicted_risk"] = 0.20

    truth = load_truth(str(p.parent / "maintenance_truth.csv"), int(time.time() // 15))
    if truth is not None:
        idx = pd.MultiIndex.from_arrays([df["vehicle_id"], df["snapshot_date"].astype(str)])
        df["Need_Maintenance"] = truth.reindex(idx).to_numpy()

    return df, total_lines

# --- الشريط الجانبي ---
st.sidebar.title("🎛️ Control Center")
refresh_sec = st.sidebar.slider("سرعة التحديث (ثواني)", 1, 5, 2)
risk_thr = st.sidebar.slider("عتبة التنبيه (Threshold)", 0.30, 0.90, 0.50, 0.05)
crit_thr = st.sidebar.slider("المستوى الحرج (Critical)", 0.50, 0.99, 0.75, 0.01)
auto_refresh = st.sidebar.toggle("بث حي مستمر", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 تصدير التقارير")

# --- الترويسة ---
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown('<p class="header-title">🚚 AI Fleet Control & Predictive Center</p>', unsafe_allow_html=True)
    st.caption("مركز المراقبة والتحليل المباشر وتقييم دقة النماذج للأسطول")

with c2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'''
        <div style="text-align:right;">
            <div class="live-indicator">
                <div class="pulse-dot"></div>
                <span>STREAMING LIVE ({refresh_sec}s)</span>
            </div>
        </div>
    ''', unsafe_allow_html=True)

st.markdown("---")

# 6. جلب وتقسيم البيانات
df_data, total_events_count = process_live_stream()

if df_data is None or df_data.empty:
    st.info("⌛ في انتظار استلام بيانات من البث المباشر...")
    if auto_refresh:
        time.sleep(1)
        st.rerun()
    st.stop()

df_data["seq"] = np.arange(len(df_data))
high_risk_df = df_data[df_data["predicted_risk"] >= risk_thr]

# زر التصدير
st.sidebar.download_button(
    "📄 تحميل الشاحنات عالية المخاطر (CSV)",
    data=high_risk_df.to_csv(index=False).encode('utf-8-sig'),
    file_name="high_risk_fleet.csv",
    mime="text/csv"
)

# --- القسم الأول: المؤشرات التشغيلية الحية (KPIs) ---
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("أحداث تمت معالجتها", f"{total_events_count:,}")
with k2:
    st.metric("شاحنات تحت المراقبة", f"{df_data['vehicle_id'].nunique():,}")
with k3:
    pct = (len(high_risk_df) / len(df_data)) * 100
    st.metric("عالية المخاطر", f"{len(high_risk_df):,}", f"{pct:.1f}% من النافذة", delta_color="inverse")
with k4:
    st.metric("متوسط الخطر العام", f"{df_data['predicted_risk'].mean():.1%}")
with k5:
    st.metric("أعلى نسبة خطر", f"{df_data['predicted_risk'].max():.1%}")

# --- القسم الثاني: مقاييس دقة النموذج اللحظية (Evaluation Metrics) ---
if "Need_Maintenance" in df_data.columns:
    m_df = df_data.dropna(subset=["Need_Maintenance"])
    if not m_df.empty:
        y_true = m_df["Need_Maintenance"].astype(int)
        y_pred = (m_df["predicted_risk"] >= risk_thr).astype(int)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-9)
        accuracy = (tp + tn) / len(m_df)

        st.markdown("<br>", unsafe_allow_html=True)
        title("🎯 مقاييس دقة وتقييم نموذج الذكاء الاصطناعي اللحظية (Model Performance)")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Accuracy (الدقة الإجمالية)", f"{accuracy:.1%}")
        q2.metric("Precision (معدل الضبط)", f"{precision:.1%}")
        q3.metric("Recall (معدل الاستدعاء)", f"{recall:.1%}")
        q4.metric("F1-Score (توازن النموذج)", f"{f1_score:.1%}")

st.markdown("<br>", unsafe_allow_html=True)

# --- القسم الثالث: شريط الأحداث الحي (Tape) + المؤشر الدائري (Gauge) ---
col_tape, col_gauge = st.columns([2.2, 1])

with col_tape:
    title("📡 شريط الأحداث الحي (Live Event Tape)")
    tape_data = df_data.tail(250)
    roll_avg = tape_data["predicted_risk"].rolling(15, min_periods=1).mean()

    fig_tape = go.Figure()
    fig_tape.add_trace(go.Scatter(
        x=tape_data["seq"], y=tape_data["predicted_risk"],
        mode="lines+markers", name="Events",
        line=dict(color="rgba(120,160,255,.3)", width=1),
        marker=dict(size=5, color=tape_data["predicted_risk"], colorscale=RISK_SCALE, cmin=0, cmax=1),
        hovertemplate="Vehicle %{text}<br>Risk %{y:.1%}<extra></extra>",
        text=tape_data["vehicle_id"]
    ))
    fig_tape.add_trace(go.Scatter(
        x=tape_data["seq"], y=roll_avg,
        mode="lines", name="Moving Avg",
        line=dict(color="#00d2ff", width=3, shape="spline")
    ))
    fig_tape.add_hline(y=risk_thr, line_dash="dash", line_color="#ffd600", annotation_text="Threshold")
    fig_tape.add_hline(y=crit_thr, line_dash="dot", line_color="#ff1744", annotation_text="Critical")
    style_fig(fig_tape, height=340, yaxis=dict(range=[0, 1.02], tickformat=".0%"))
    show_chart(fig_tape, "event_tape")

with col_gauge:
    title("🎯 مؤشر خطورة الأسطول")
    gauge_val = df_data["predicted_risk"].mean() * 100

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_val,
        number=dict(suffix="%", font=dict(size=40, color="#ffffff")),
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00d2ff", 'thickness': 0.25},
            'steps': [
                {'range': [0, 40], 'color': "#1b5e20"},
                {'range': [40, 70], 'color': "#f57f17"},
                {'range': [70, 100], 'color': "#b71c1c"}
            ],
            'threshold': {'line': {'color': "white", 'width': 3}, 'value': risk_thr * 100}
        }
    ))
    style_fig(fig_gauge, height=340)
    show_chart(fig_gauge, "fleet_gauge")

st.markdown("<br>", unsafe_allow_html=True)

# --- القسم الرابع: توزيع المخاطر حسب فئات الشاحنات (DRY_VAN, REEFER, FLATBED...) + سباق الشاحنات ---
col_pulse, col_race = st.columns([1.5, 1])

with col_pulse:
    title("📊 توزيع المخاطر والأنواع حياً (DRY_VAN, REEFER, TANKER...)")
    if "vehicle_type" in df_data.columns:
        type_summary = df_data.groupby("vehicle_type").agg(
            avg_risk=("predicted_risk", "mean"),
            high_count=("predicted_risk", lambda x: (x >= risk_thr).sum())
        ).reset_index()

        fig_type_bar = px.bar(
            type_summary,
            x="vehicle_type",
            y="avg_risk",
            color="high_count",
            color_continuous_scale="Reds",
            text_auto=".1%",
            labels={"vehicle_type": "نوع الشاحنة", "avg_risk": "متوسط الخطر", "high_count": "عدد الحالات الحرجة"}
        )
        fig_type_bar.update_traces(textposition="outside")
        style_fig(fig_type_bar, height=360, yaxis=dict(tickformat=".0%", range=[0, max(0.4, type_summary['avg_risk'].max() * 1.25)]))
        show_chart(fig_type_bar, "vehicle_type_live_bar")

with col_race:
    title("🏁 سباق أخطر 12 شاحنة (Live Ranking)")
    latest_vehicles = df_data.sort_values("seq").drop_duplicates("vehicle_id", keep="last")
    top12 = latest_vehicles.sort_values("predicted_risk", ascending=False).head(12).sort_values("predicted_risk")

    fig_race = go.Figure(go.Bar(
        x=top12["predicted_risk"], y=top12["vehicle_id"],
        orientation="h",
        text=[f"{v:.1%}" for v in top12["predicted_risk"]],
        textposition="inside",
        marker=dict(color=top12["predicted_risk"], colorscale=RISK_SCALE, cmin=0, cmax=1)
    ))
    fig_race.add_vline(x=risk_thr, line_dash="dash", line_color="white")
    style_fig(fig_race, height=360, xaxis=dict(range=[0, 1], tickformat=".0%"))
    show_chart(fig_race, "live_ranking")

st.markdown("<br>", unsafe_allow_html=True)

# --- القسم الخامس: التحليل 3D التفاعلي + شريط التنبيهات السريع (Incident Feed) ---
col_3d, col_feed = st.columns([1.8, 1])

with col_3d:
    title("🌐 التحليل ثلاثي الأبعاد (3D Fleet Risk Mapping)")
    sample_3d = df_data.sample(min(800, len(df_data)))
    
    fig_3d = px.scatter_3d(
        sample_3d,
        x='odometer_km',
        y='predicted_risk',
        z='brake_condition' if 'brake_condition' in df_data.columns else 'predicted_risk',
        color='predicted_risk',
        color_continuous_scale=RISK_SCALE,
        hover_data=['vehicle_id', 'vehicle_type']
    )
    fig_3d.update_traces(marker=dict(size=4.5, opacity=0.85))
    style_fig(
        fig_3d,
        height=400,
        scene=dict(
            xaxis_title="العداد (كم)",
            yaxis_title="نسبة الخطر",
            zaxis_title="حالة الفرامل",
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.0))
        )
    )
    show_chart(fig_3d, "scatter_3d")

with col_feed:
    title("⚡ آخر الأحداث الواردة (Incident Feed)")
    recent_feed = df_data.tail(7).iloc[::-1]
    feed_html = ""
    for r in recent_feed.itertuples():
        rk = r.predicted_risk
        color_code = "#ff1744" if rk >= crit_thr else "#ff9100" if rk >= risk_thr else "#00e676"
        vt = getattr(r, "vehicle_type", "TRUCK")
        feed_html += f'''
        <div class="fc" style="border-left-color: {color_code};">
            <div>
                <b>{r.vehicle_id}</b><br>
                <small>{vt} • {r.odometer_km:,.0f} KM</small>
            </div>
            <div class="r" style="color: {color_code};">{rk:.1%}</div>
        </div>
        '''
    st.markdown(feed_html, unsafe_allow_html=True)

# --- القسم السادس: جدول التنبيهات المباشرة ---
st.markdown("---")
title("🚨 جدول الشاحنات الأكثر خطورة والمتطلبة للصيانة الفورية")

top_table = latest_vehicles[latest_vehicles["predicted_risk"] >= risk_thr].sort_values("predicted_risk", ascending=False).head(30).copy()

if not top_table.empty:
    top_table["Risk %"] = top_table["predicted_risk"] * 100
    top_table["snapshot_date"] = pd.to_datetime(top_table["snapshot_date"]).dt.strftime("%Y-%m-%d")
    cols_show = [c for c in ["vehicle_id", "snapshot_date", "vehicle_type", "odometer_km", "Risk %", "Need_Maintenance"] if c in top_table.columns]

    st.dataframe(
        top_table[cols_show],
        hide_index=True,
        height=320,
        **DF_W,
        column_config={
            "Risk %": st.column_config.ProgressColumn("نسبة الخطر %", min_value=0, max_value=100, format="%.1f%%"),
            "odometer_km": st.column_config.NumberColumn("العداد (كم)", format="%d"),
            "vehicle_id": "رقم الشاحنة",
            "snapshot_date": "تاريخ القراءة",
            "vehicle_type": "نوع الشاحنة",
            "Need_Maintenance": "الحقيقة (Actual)"
        }
    )
else:
    st.success("✅ جميع الشاحنات في حالة سليمة حالياً.")

# --- التحديث دون وميض ---
frag = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()