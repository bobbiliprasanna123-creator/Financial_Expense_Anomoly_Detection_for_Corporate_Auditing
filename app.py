import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Expense Intelligence | Audit Analytics", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
.stApp { background-color: #F1F5F9; }
.block-container { padding: 2rem 2.5rem; max-width: 1400px; }
[data-testid="stSidebar"] { background-color: #0F172A; border-right: 1px solid #1E293B; }
[data-testid="stSidebar"] > div { padding-top: 0px; }
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] hr { border-color: #1E293B !important; margin: 12px 0; }
div[data-testid="metric-container"] { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
div[data-testid="metric-container"] label { font-size: 0.72rem !important; font-weight: 600 !important; color: #64748B !important; text-transform: uppercase; letter-spacing: 0.07em; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { font-size: 1.75rem !important; font-weight: 700 !important; color: #0F172A !important; }
.stTabs [data-baseweb="tab-list"] { background-color: #FFFFFF; border-radius: 8px 8px 0 0; border-bottom: 2px solid #E2E8F0; padding: 0 16px; gap: 0; }
.stTabs [data-baseweb="tab"] { font-size: 0.9rem; font-weight: 600; color: #64748B; padding: 14px 24px; background: transparent; border: none; border-bottom: 2px solid transparent; margin-bottom: -2px; }
.stTabs [aria-selected="true"] { color: #1E40AF !important; border-bottom: 2px solid #1E40AF !important; }
.stFormSubmitButton > button { background-color: #1E40AF !important; color: white !important; border-radius: 6px !important; font-weight: 600 !important; font-size: 0.95rem !important; padding: 12px 20px !important; border: none !important; width: 100%; }
[data-testid="stDataFrame"] { border-radius: 8px; border: 1px solid #E2E8F0; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

FRIENDLY_NAMES = {
    "amount":                  ("Transaction Amount",                "The raw transaction value submitted for reimbursement."),
    "expense_category":        ("Expense Category",                  "The category under which the expense was filed."),
    "department":              ("Department",                         "The employee department at time of submission."),
    "employee_role":           ("Employee Role",                      "The employee organisational role."),
    "payment_method":          ("Payment Method",                     "The method used to settle the expense."),
    "hour":                    ("Submission Time",                    "The hour of day when the expense was submitted."),
    "employee_category_avg":   ("Employee Historical Average",        "The employee average spend in this category over time."),
    "category_avg_amount":     ("Company Category Benchmark",         "The company-wide average amount for this expense category."),
    "monthly_employee_spend":  ("Month-to-Date Employee Spend",       "Total employee expenditure recorded for the current month."),
    "transaction_frequency":   ("Transaction Frequency",              "Number of transactions submitted by this employee this month."),
    "is_weekend":              ("Weekend Submission",                  "Whether the expense was submitted on a weekend."),
    "amount_to_emp_cat_ratio": ("Spend vs. Employee Baseline",        "How much this transaction deviates from the employee own historical average for this category."),
    "amount_to_cat_avg_ratio": ("Spend vs. Company Benchmark",        "How far this amount is above the company-wide benchmark for this expense category."),
    "mtd_spend_share":         ("Share of Monthly Budget",            "This transaction as a proportion of the employee total month-to-date expenditure."),
    "log_amount":              ("Normalised Transaction Amount",       "Logarithmically scaled amount used for model calibration."),
    "is_off_hours":            ("Off-Hours Submission (1AM - 4AM)",   "Expense submitted during an unusual late-night window associated with elevated risk."),
    "is_late_night":           ("Late-Night Submission",              "Expense submitted after 10 PM or before 5 AM."),
    "is_month_end":            ("Month-End Submission",               "Expense submitted in the final days of the month, a common period for irregular activity."),
    "is_smurfing_band":        ("Amount Near Approval Threshold",     "Transaction amount falls just below the 10,000 threshold, a pattern associated with threshold avoidance."),
    "is_round_number":         ("Round-Number Transaction",           "Transaction uses a suspiciously round amount, which can indicate estimate-based or fabricated claims."),
    "is_reimbursement":        ("Out-of-Pocket Reimbursement",        "Employee paid personally and is seeking reimbursement, which is a higher-risk payment path."),
    "high_velocity_flag":      ("Elevated Transaction Volume",        "The employee has submitted an unusually high number of transactions this month."),
    "modified_z_score":        ("Statistical Deviation from Norm",   "A robust statistical measure of how far this transaction sits from the expected spending range for this category."),
    "vendor":                  ("Vendor / Merchant",                  "The merchant or service provider where the expense was incurred."),
    "business_purpose":        ("Business Purpose",                   "The declared business justification or purpose code."),
    "approval_status":         ("Approval Status",                    "The workflow approval state or pre-clearance authorization."),
    "day_of_week":             ("Day of Week",                        "The day of the week the transaction occurred (0=Monday to 6=Sunday)."),
    "day_of_month":            ("Day of Month",                       "The calendar day of the month when the expense occurred.")
}

@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open(BASE_DIR / "best_anomaly_model.pkl", "rb") as f: model = pickle.load(f)
    with open(BASE_DIR / "threshold_config.json", "r") as f: threshold_cfg = json.load(f)
    with open(BASE_DIR / "feature_columns.json", "r") as f: feature_cfg = json.load(f)
    with open(BASE_DIR / "category_encodings.json", "r") as f: encodings = json.load(f)
    with open(BASE_DIR / "baseline_stats.json", "r") as f: baselines = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, threshold_cfg, feature_cfg, encodings, baselines, explainer

model, threshold_cfg, feature_cfg, encodings, baselines, explainer = load_artifacts()
THRESHOLD       = threshold_cfg["optimal_threshold"]
FEATURE_COLUMNS = feature_cfg["feature_columns"]
CAT_MEDIANS     = baselines["category_medians"]
CAT_MADS        = baselines["category_mads"]

def encode_categorical(value, col): 
    return encodings.get(col, {}).get("code_map", {}).get(str(value), 0)

def assign_priority_tier(score):
    if   score >= 0.85: return "Tier 1 - Critical",  "CRITICAL"
    elif score >= 0.70: return "Tier 2 - High Risk",  "HIGH"
    elif score >= 0.55: return "Tier 3 - Moderate",   "MEDIUM"
    else:               return "Tier 4 - Low Risk",   "LOW"

def engineer_features(row):
    amt = float(row["amount"])
    feat = {
        "employee_role":          encode_categorical(row.get("employee_role","Analyst"),"employee_role"),
        "department":             encode_categorical(row.get("department","Operations"),"department"),
        "expense_category":       encode_categorical(row.get("expense_category","Food"),"expense_category"),
        "amount":                 amt,
        "payment_method":         encode_categorical(row.get("payment_method","Corporate Card"),"payment_method"),
        "vendor":                 encode_categorical(row.get("vendor","Unknown"),"vendor"),
        "business_purpose":       encode_categorical(row.get("business_purpose","General"),"business_purpose"),
        "approval_status":        encode_categorical(row.get("approval_status","Approved"),"approval_status"),
        "monthly_employee_spend": float(row.get("monthly_employee_spend",0)),
        "employee_category_avg":  float(row.get("employee_category_avg",amt)),
        "category_avg_amount":    float(row.get("category_avg_amount",amt)),
        "transaction_frequency":  int(row.get("transaction_frequency",0)),
        "day_of_week":            int(row.get("day_of_week",0)),
        "is_weekend":             int(row.get("is_weekend",0)),
        "hour":                   int(row.get("hour",12)),
        "amount_to_emp_cat_ratio":round(amt/(float(row.get("employee_category_avg",amt))+1e-5),4),
        "amount_to_cat_avg_ratio":round(amt/(float(row.get("category_avg_amount",amt))+1e-5),4),
        "mtd_spend_share":        round(amt/(float(row.get("monthly_employee_spend",0))+amt+1e-5),4),
        "log_amount":             round(np.log1p(amt),4),
        "is_off_hours":           int(int(row.get("hour",12)) in [1,2,3,4]),
        "is_late_night":          int(int(row.get("hour",12))>=22 or int(row.get("hour",12))<=5),
        "day_of_month":           int(row.get("day_of_month",15)),
        "is_month_end":           int(int(row.get("day_of_month",15))>=26),
        "is_smurfing_band":       int(9500<=amt<=9999),
        "is_round_number":        int(amt%1000==0 and amt>=5000),
        "is_reimbursement":       int(str(row.get("payment_method"))=="Reimbursement"),
        "high_velocity_flag":     int(int(row.get("transaction_frequency",0))>=12),
        "modified_z_score":       round(0.6745*(amt-CAT_MEDIANS.get(str(row.get("expense_category","Food")),amt))/(CAT_MADS.get(str(row.get("expense_category","Food")),1.0)+1e-5),4),
    }
    df = pd.DataFrame([feat])
    for col in FEATURE_COLUMNS:
        if col not in df.columns: df[col] = 0
    return df[FEATURE_COLUMNS]

def predict(feature_df):
    proba     = model.predict_proba(feature_df)[:,1][0]
    flag      = int(proba>=THRESHOLD)
    z_raw     = max(float(feature_df["modified_z_score"].values[0]), 0.0)
    z_norm    = 1.0 - np.exp(-z_raw / 20.0)
    composite = round(0.50*proba+0.25*z_norm+0.15*float(feature_df["high_velocity_flag"].values[0])+0.10*float(feature_df["is_off_hours"].values[0]),4)
    return proba, flag, composite

def format_probability(p):
    if pd.isna(p): return "-"
    if p > 0.99: return ">99%"
    if p < 0.01: return "<1%"
    return f"{p*100:.1f}%"

def chart_layout(fig, title, height=320):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#1E293B", family="Inter"), x=0, xanchor="left"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=height,
        margin=dict(t=40, b=12, l=0, r=0),
        font=dict(family="Inter", color="#475569", size=11),
        legend=dict(font=dict(size=10)),
        hoverlabel=dict(bgcolor="white", font_size=11, font_family="Inter")
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False, tickfont=dict(size=10))
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False, tickfont=dict(size=10))
    return fig

def divider(): 
    st.markdown("<hr style='border:none;border-top:1px solid #E2E8F0;margin:20px 0;'>", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("<div style='background:#1E293B;margin:-1rem -1rem 0 -1rem;padding:28px 24px 20px 24px;border-bottom:1px solid #334155;'><div style='font-size:1.2rem;font-weight:700;color:#F8FAFC;line-height:1.2;'>EXPENSE<br>INTELLIGENCE</div><div style='font-size:0.7rem;color:#94A3B8;margin-top:6px;font-weight:500;text-transform:uppercase;letter-spacing:0.08em;'>Audit Risk Prioritization</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.68rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.1em;padding:0 0 8px 0;'>System Configuration</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1E293B;margin:0 0 12px 0;'>", unsafe_allow_html=True)
    cfg1, cfg2 = st.columns(2)
    with cfg1:
        st.markdown(f"<div style='background:#1E293B;border-radius:6px;padding:12px;text-align:center;'><div style='font-size:0.62rem;color:#64748B;text-transform:uppercase;letter-spacing:0.05em;font-weight:600;'>Threshold</div><div style='font-size:1.25rem;font-weight:700;color:#F8FAFC;margin-top:2px;'>{THRESHOLD:.2f}</div></div>", unsafe_allow_html=True)
    with cfg2:
        st.markdown(f"<div style='background:#1E293B;border-radius:6px;padding:12px;text-align:center;'><div style='font-size:0.62rem;color:#64748B;text-transform:uppercase;letter-spacing:0.05em;font-weight:600;'>Precision</div><div style='font-size:1.25rem;font-weight:700;color:#F8FAFC;margin-top:2px;'>{threshold_cfg.get('precision_at_threshold',0)*100:.1f}%</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.68rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.1em;padding:0 0 8px 0;'>Risk Routing Protocol</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1E293B;margin:0 0 8px 0;'>", unsafe_allow_html=True)
    for color, tier, label, rule in [
        ("#EF4444", "Tier 1", "Critical Risk", "Score >= 0.85"),
        ("#F97316", "Tier 2", "High Risk", "Score >= 0.70"),
        ("#EAB308", "Tier 3", "Moderate Risk", "Score >= 0.55"),
        ("#22C55E", "Tier 4", "Low Risk", "Score < 0.55")
    ]:
        st.markdown(f"<div style='display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid #1E293B;'><div style='width:3px;height:30px;background:{color};border-radius:2px;flex-shrink:0;'></div><div><div style='font-size:0.78rem;font-weight:600;color:#E2E8F0;'>{tier} - {label}</div><div style='font-size:0.68rem;color:#64748B;'>{rule}</div></div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='background:#1E293B;border-radius:6px;padding:10px 14px;display:flex;align-items:center;gap:8px;'><div style='width:8px;height:8px;background:#22C55E;border-radius:50%;flex-shrink:0;'></div><div><span style='font-size:0.72rem;color:#94A3B8;font-weight:600;'>MODEL STATUS</span><span style='font-size:0.72rem;color:#22C55E;font-weight:700;margin-left:6px;'>Operational</span></div></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.62rem;color:#334155;margin-top:16px;text-align:center;'>Audit Analytics Platform v2.0</div>", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:20px;'><div style='font-size:1.55rem;font-weight:700;color:#0F172A;letter-spacing:-0.02em;'>Expense Intelligence Platform</div><div style='font-size:0.87rem;color:#64748B;margin-top:3px;'>AI-assisted expense risk prioritization and audit analytics</div></div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["  Overview Dashboard  ", "  Transaction Risk Analysis  "])

with tab1:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:22px 24px 16px 24px;margin-bottom:16px;'><div style='font-size:0.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;'>Data Import</div><div style='font-size:0.98rem;font-weight:700;color:#0F172A;margin-bottom:3px;'>Import Corporate Expense Ledger</div><div style='font-size:0.82rem;color:#64748B;'>Upload a CSV export of expense transactions to analyse risk levels and generate the prioritized audit queue.</div></div>", unsafe_allow_html=True)
    
    # Template CSV generator
    sample_template = pd.DataFrame([
        {
            "expense_id": "EXP-1001",
            "employee_id": "EMP-5012",
            "department": "Sales",
            "employee_role": "Sales Executive",
            "expense_category": "Client Meeting",
            "amount": 3200.00,
            "payment_method": "Corporate Card",
            "vendor": "Marriott Dining",
            "business_purpose": "Quarterly Client Review",
            "approval_status": "Approved",
            "hour": 19,
            "day_of_month": 15,
            "day_of_week": 2,
            "is_weekend": 0,
            "employee_category_avg": 3100.00,
            "category_avg_amount": 3160.51,
            "monthly_employee_spend": 18500.00,
            "transaction_frequency": 8
        },
        {
            "expense_id": "EXP-1002",
            "employee_id": "EMP-2041",
            "department": "Engineering",
            "employee_role": "Software Engineer",
            "expense_category": "Food",
            "amount": 450.00,
            "payment_method": "Corporate Card",
            "vendor": "Subway",
            "business_purpose": "Team Lunch",
            "approval_status": "Approved",
            "hour": 13,
            "day_of_month": 12,
            "day_of_week": 3,
            "is_weekend": 0,
            "employee_category_avg": 520.00,
            "category_avg_amount": 859.02,
            "monthly_employee_spend": 3200.00,
            "transaction_frequency": 6
        }
    ])
    
    up_c1, up_c2 = st.columns([3, 1])
    with up_c1:
        uploaded_file = st.file_uploader(
            "Drop CSV file here or browse to upload",
            type=["csv"],
            help="Required columns: amount, expense_category, department, employee_role, payment_method, hour, day_of_month, is_weekend, employee_category_avg, category_avg_amount, monthly_employee_spend, transaction_frequency"
        )
    with up_c2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.download_button(
            "Download sample CSV template",
            data=sample_template.to_csv(index=False).encode("utf-8"),
            file_name="expense_ledger_template.csv",
            mime="text/csv",
            help="Download a clean template with the required column structure."
        )

    REQUIRED_COLS = ["amount", "expense_category", "department", "employee_role", "payment_method", "hour", "day_of_month", "is_weekend", "employee_category_avg", "category_avg_amount", "monthly_employee_spend", "transaction_frequency"]

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Unable to read uploaded CSV file: {e}. Please ensure the file is valid UTF-8 CSV data.")
            st.stop()

        if df_upload.empty:
            st.warning("The uploaded CSV file is empty. Please upload a file containing transaction records.")
            st.stop()

        missing = [c for c in REQUIRED_COLS if c not in df_upload.columns]
        if missing:
            st.error(f"Validation failed. The uploaded file is missing **{len(missing)}** required column(s): **{', '.join(missing)}**. Please review and re-upload.")
            st.stop()

        fc1, fc2, fc3 = st.columns(3)
        fc1.markdown(f"<div style='background:#F0FDF4;border:1px solid #BBF7D0;border-radius:6px;padding:10px 14px;'><span style='font-size:0.72rem;color:#15803D;font-weight:700;'>FILE LOADED</span><br><span style='font-size:0.82rem;color:#0F172A;font-weight:600;'>{uploaded_file.name}</span></div>", unsafe_allow_html=True)
        fc2.markdown(f"<div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:6px;padding:10px 14px;'><span style='font-size:0.72rem;color:#1D4ED8;font-weight:700;'>ROWS LOADED</span><br><span style='font-size:0.82rem;color:#0F172A;font-weight:600;'>{len(df_upload):,} transactions</span></div>", unsafe_allow_html=True)
        fc3.markdown(f"<div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:6px;padding:10px 14px;'><span style='font-size:0.72rem;color:#1D4ED8;font-weight:700;'>COLUMNS DETECTED</span><br><span style='font-size:0.82rem;color:#0F172A;font-weight:600;'>{len(df_upload.columns)} columns &nbsp;<span style='color:#15803D;'>Valid</span></span></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        with st.spinner("Running risk assessment engine across all transactions..."):
            results = []
            errors = []
            for idx, row in df_upload.iterrows():
                try:
                    feat_df = engineer_features(row.to_dict())
                    proba, flag, comp = predict(feat_df)
                    tier_label, urgency = assign_priority_tier(comp)
                    results.append({"Risk_Probability": proba, "Audit_Score": comp, "Flag": flag, "Priority": tier_label, "Urgency": urgency})
                except Exception as e:
                    errors.append((idx, str(e)))
                    results.append({"Risk_Probability": None, "Audit_Score": None, "Flag": 0, "Priority": "Tier 4 - Low Risk", "Urgency": "LOW"})
            if errors:
                st.warning(f"{len(errors)} transaction(s) could not be fully scored. First issue at row {errors[0][0]}: {errors[0][1]}")

        results_df = pd.DataFrame(results)
        audit_df = pd.concat([df_upload.reset_index(drop=True), results_df], axis=1)
        flagged_df = audit_df[audit_df["Flag"] == 1].sort_values("Audit_Score", ascending=False).reset_index(drop=True)
        total_exp = audit_df["amount"].sum() if "amount" in audit_df.columns else 0
        total_analyzed = len(audit_df)
        flag_rate = (len(flagged_df) / total_analyzed * 100) if total_analyzed > 0 else 0

        divider()
        st.markdown("<div style='font-size:0.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;'>Executive Summary</div>", unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Expenses", f"Rs. {total_exp/1_000_000:.2f}M" if total_exp > 1_000_000 else f"Rs. {total_exp:,.0f}")
        k2.metric("Transactions Analysed", f"{total_analyzed:,}")
        k3.metric("High-Risk Transactions", f"{len(flagged_df):,}")
        k4.metric(
            "Escalation Rate",
            f"{flag_rate:.1f}%",
            help="Percentage of total transactions analysed flagged for prioritized audit review (High-Risk Transactions ÷ Total Transactions Analysed)."
        )
        
        divider()
        color_map = {
            "Tier 1 - Critical": "#EF4444",
            "Tier 2 - High Risk": "#F97316",
            "Tier 3 - Moderate": "#EAB308",
            "Tier 4 - Low Risk": "#22C55E"
        }

        # Score to tier legend banner
        st.markdown("""
        <div style='background:#FFFFFF;border:1px solid #E2E8F0;border-radius:6px;padding:12px 18px;margin-bottom:16px;display:flex;flex-wrap:wrap;gap:20px;align-items:center;'>
            <span style='font-size:0.72rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.06em;'>Score-to-Tier Mapping Legend:</span>
            <span style='font-size:0.8rem;color:#1E293B;'><strong style='color:#EF4444;'>Tier 1 - Critical</strong> (Score &ge; 0.85)</span>
            <span style='font-size:0.8rem;color:#1E293B;'><strong style='color:#F97316;'>Tier 2 - High Risk</strong> (Score &ge; 0.70)</span>
            <span style='font-size:0.8rem;color:#1E293B;'><strong style='color:#EAB308;'>Tier 3 - Moderate</strong> (Score &ge; 0.55)</span>
            <span style='font-size:0.8rem;color:#1E293B;'><strong style='color:#22C55E;'>Tier 4 - Low Risk</strong> (Score &lt; 0.55)</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-size:0.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;'>Risk Overview</div>", unsafe_allow_html=True)
        ro1, ro2 = st.columns(2)
        with ro1:
            tier_order = ["Tier 1 - Critical", "Tier 2 - High Risk", "Tier 3 - Moderate", "Tier 4 - Low Risk"]
            tier_counts = audit_df["Priority"].value_counts().reindex(tier_order, fill_value=0).reset_index()
            tier_counts.columns = ["Priority", "Count"]
            fig_donut = px.pie(
                tier_counts,
                names="Priority",
                values="Count",
                hole=0.60,
                color="Priority",
                color_discrete_map=color_map
            )
            fig_donut = chart_layout(fig_donut, "Risk Distribution by Tier (All Transactions)")
            fig_donut.update_traces(textposition="outside", textinfo="percent+label", textfont_size=10, marker=dict(line=dict(color="white", width=2)), sort=False)
            st.plotly_chart(fig_donut, use_container_width=True)
        with ro2:
            st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
            for tname in tier_order:
                count = (audit_df["Priority"] == tname).sum()
                pct = (count / total_analyzed * 100) if total_analyzed > 0 else 0
                color = color_map.get(tname, "#94A3B8")
                st.markdown(f"""
                <div style='margin-bottom:12px;'>
                    <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                        <span style='font-size:0.82rem;color:#1E293B;font-weight:600;'>{tname}</span>
                        <span style='font-size:0.82rem;color:#64748B;'>{count:,} ({pct:.1f}%)</span>
                    </div>
                    <div style='background:#F1F5F9;border-radius:4px;height:7px;overflow:hidden;'>
                        <div style='background:{color};width:{pct}%;height:7px;border-radius:4px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        divider()
        st.markdown("<div style='font-size:0.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;'>Expense Analytics</div>", unsafe_allow_html=True)
        ea1, ea2 = st.columns(2)
        with ea1:
            if "department" in flagged_df.columns and len(flagged_df) > 0:
                dept_data = flagged_df.groupby("department").agg(Count=("Audit_Score","count")).reset_index().sort_values("Count", ascending=True).tail(8)
                fig_dept = px.bar(
                    dept_data,
                    x="Count",
                    y="department",
                    orientation="h",
                    color_discrete_sequence=["#1E40AF"],
                    labels={"department": "Department", "Count": "Flagged Transactions"}
                )
                fig_dept = chart_layout(fig_dept, "High-Risk Transactions by Department")
                fig_dept.update_xaxes(dtick=1, tickformat="d")
                st.plotly_chart(fig_dept, use_container_width=True)
            else:
                st.info("No department risk data to display.")
        with ea2:
            if "expense_category" in flagged_df.columns and len(flagged_df) > 0:
                cat_data = flagged_df.groupby("expense_category").agg(Count=("Audit_Score","count"), Total=("amount","sum")).reset_index().sort_values("Total", ascending=True).tail(8)
                fig_cat = px.bar(cat_data, x="Total", y="expense_category", orientation="h", color_discrete_sequence=["#7C3AED"], labels={"expense_category": "Category", "Total": "Total Risk Exposure (Rs.)"})
                fig_cat = chart_layout(fig_cat, "Risk Exposure by Expense Category")
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("No category risk exposure data to display.")
        
        ea3, ea4 = st.columns(2)
        with ea3:
            if "day_of_month" in audit_df.columns:
                time_data = audit_df.groupby("day_of_month").agg(Total=("Flag","sum")).reset_index()
                fig_time = px.bar(
                    time_data,
                    x="day_of_month",
                    y="Total",
                    color_discrete_sequence=["#1E40AF"],
                    labels={"day_of_month": "Day of Month", "Total": "Flagged Transactions"}
                )
                fig_time = chart_layout(fig_time, "High-Risk Transactions by Day of Month")
                fig_time.update_xaxes(dtick=2, tickformat="d")
                fig_time.update_yaxes(dtick=1, tickformat="d")
                st.plotly_chart(fig_time, use_container_width=True)
        with ea4:
            if "expense_category" in audit_df.columns and "amount" in audit_df.columns:
                avg_amt = audit_df.groupby("expense_category")["amount"].mean().reset_index().sort_values("amount", ascending=False).head(8)
                fig_avg = px.bar(avg_amt, x="expense_category", y="amount", color_discrete_sequence=["#0EA5E9"], labels={"expense_category": "", "amount": "Average Amount (Rs.)"})
                fig_avg = chart_layout(fig_avg, "Average Expense Amount by Category")
                fig_avg.update_xaxes(tickangle=-30)
                st.plotly_chart(fig_avg, use_container_width=True)

        divider()
        st.markdown("<div style='font-size:0.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>Prioritized Audit Queue</div>", unsafe_allow_html=True)
        
        if len(flagged_df) > 0:
            # Initialize persistent review status dictionary in session_state
            if "review_status" not in st.session_state:
                st.session_state["review_status"] = {}

            # Filters container
            with st.container():
                st.markdown("<div style='font-size:0.75rem;font-weight:600;color:#64748B;margin-bottom:6px;'>FILTER QUEUE</div>", unsafe_allow_html=True)
                f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1.4])
                with f1:
                    avail_depts = sorted(flagged_df["department"].dropna().unique().tolist()) if "department" in flagged_df.columns else []
                    sel_depts = st.multiselect("Department", options=avail_depts, default=avail_depts)
                with f2:
                    avail_cats = sorted(flagged_df["expense_category"].dropna().unique().tolist()) if "expense_category" in flagged_df.columns else []
                    sel_cats = st.multiselect("Expense Category", options=avail_cats, default=avail_cats)
                with f3:
                    avail_tiers = [t for t in tier_order if t in flagged_df["Priority"].values]
                    sel_tiers = st.multiselect("Risk Tier", options=avail_tiers, default=avail_tiers)
                with f4:
                    top_k = st.slider("Top K Transactions", min_value=1, max_value=max(1, len(flagged_df)), value=min(50, len(flagged_df)), step=1)

            # Apply filters
            filtered_queue = flagged_df.copy()
            if sel_depts and "department" in filtered_queue.columns:
                filtered_queue = filtered_queue[filtered_queue["department"].isin(sel_depts)]
            if sel_cats and "expense_category" in filtered_queue.columns:
                filtered_queue = filtered_queue[filtered_queue["expense_category"].isin(sel_cats)]
            if sel_tiers:
                filtered_queue = filtered_queue[filtered_queue["Priority"].isin(sel_tiers)]
            filtered_queue = filtered_queue.head(top_k).reset_index(drop=True)

            if len(filtered_queue) == 0:
                st.info("No transactions match the selected filter criteria.")
            else:
                # Build display dataframe
                queue_records = []
                for rank_idx, (_, r) in enumerate(filtered_queue.iterrows(), start=1):
                    txn_id = str(r["expense_id"]) if "expense_id" in r and pd.notna(r["expense_id"]) else f"TXN-{rank_idx:04d}"
                    rec = {
                        "Rank": rank_idx,
                        "Transaction ID": txn_id,
                        "Category": r.get("expense_category", "-"),
                        "Department": r.get("department", "-"),
                        "Amount (Rs.)": float(r.get("amount", 0.0)),
                        "Risk Score": float(r.get("Audit_Score", 0.0)),
                        "Risk Tier": r.get("Priority", "-"),
                        "Model Risk Probability": format_probability(r.get("Risk_Probability")),
                        "Review Status": st.session_state["review_status"].get(txn_id, "Pending")
                    }
                    if "employee_id" in r:
                        rec["Employee ID"] = str(r["employee_id"])
                    queue_records.append(rec)
                
                queue_display_df = pd.DataFrame(queue_records)
                
                # Dynamic height calculation to eliminate empty white space
                table_height = min(460, max(140, (len(queue_display_df) + 1) * 36 + 40))

                # Interactive editable table with Review Status
                edited_queue = st.data_editor(
                    queue_display_df,
                    column_config={
                        "Review Status": st.column_config.SelectboxColumn(
                            "Review Status",
                            help="Update audit workflow decision (Pending / Confirmed / Dismissed)",
                            options=["Pending", "Confirmed", "Dismissed"],
                            required=True,
                            default="Pending"
                        ),
                        "Rank": st.column_config.NumberColumn("Rank", disabled=True),
                        "Transaction ID": st.column_config.TextColumn("Transaction ID", disabled=True),
                        "Category": st.column_config.TextColumn("Category", disabled=True),
                        "Department": st.column_config.TextColumn("Department", disabled=True),
                        "Amount (Rs.)": st.column_config.NumberColumn("Amount (Rs.)", format="Rs. %.2f", disabled=True),
                        "Risk Score": st.column_config.NumberColumn("Risk Score", format="%.3f", disabled=True),
                        "Risk Tier": st.column_config.TextColumn("Risk Tier", disabled=True),
                        "Model Risk Probability": st.column_config.TextColumn("Model Risk Probability", disabled=True),
                    },
                    disabled=[c for c in queue_display_df.columns if c != "Review Status"],
                    hide_index=True,
                    use_container_width=True,
                    height=table_height,
                    key="audit_queue_editor"
                )

                # Persist review status updates in session state
                for _, erow in edited_queue.iterrows():
                    st.session_state["review_status"][erow["Transaction ID"]] = erow["Review Status"]

                # Export CSV button including Review Status
                export_df = flagged_df.copy()
                export_df["Review_Status"] = [
                    st.session_state["review_status"].get(
                        str(r["expense_id"]) if "expense_id" in r and pd.notna(r["expense_id"]) else f"TXN-{i+1:04d}",
                        "Pending"
                    )
                    for i, r in export_df.iterrows()
                ]

                dl_c1, _ = st.columns([1, 3])
                with dl_c1:
                    st.download_button(
                        "Export Audit Queue (CSV)",
                        data=export_df.to_csv(index=False).encode("utf-8"),
                        file_name="audit_queue_export.csv",
                        mime="text/csv",
                        type="primary"
                    )

                divider()
                st.markdown("<div style='font-size:0.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;'>Auditor Deep-Dive: Transaction Explanation</div><div style='font-size:0.95rem;font-weight:700;color:#0F172A;margin-bottom:12px;'>Select a Transaction to Inspect SHAP Risk Drivers</div>", unsafe_allow_html=True)
                
                selected_txn_id = st.selectbox(
                    "Select Transaction from Queue:",
                    options=queue_display_df["Transaction ID"].tolist(),
                    format_func=lambda x: f"Rank {queue_display_df.loc[queue_display_df['Transaction ID']==x, 'Rank'].values[0]} | {x} — {queue_display_df.loc[queue_display_df['Transaction ID']==x, 'Category'].values[0]} (Rs. {queue_display_df.loc[queue_display_df['Transaction ID']==x, 'Amount (Rs.)'].values[0]:,.2f}, Score: {queue_display_df.loc[queue_display_df['Transaction ID']==x, 'Risk Score'].values[0]:.3f})"
                )

                if selected_txn_id:
                    matched_idx = None
                    for orig_idx, r in flagged_df.iterrows():
                        cur_id = str(r["expense_id"]) if "expense_id" in r and pd.notna(r["expense_id"]) else f"TXN-{orig_idx+1:04d}"
                        if cur_id == selected_txn_id:
                            matched_idx = orig_idx
                            break
                    if matched_idx is None:
                        matched_idx = 0

                    sel_row = flagged_df.iloc[matched_idx]
                    sel_feat_df = engineer_features(sel_row.to_dict())
                    sel_shap_raw = explainer.shap_values(sel_feat_df)
                    sel_shap_vals = sel_shap_raw[1][0] if isinstance(sel_shap_raw, list) else sel_shap_raw[0]

                    sh_col1, sh_col2 = st.columns([3, 2])
                    sel_impacts = [{"feature": col, "impact": sel_shap_vals[i], "value": sel_feat_df.iloc[0, i]} for i, col in enumerate(FEATURE_COLUMNS)]
                    sel_impacts_sorted = sorted(sel_impacts, key=lambda x: x["impact"], reverse=True)
                    sel_pos_drivers = [x for x in sel_impacts_sorted if x["impact"] > 0]
                    sel_top_drivers = sel_pos_drivers[:4]
                    sel_top_mitigators = [x for x in reversed(sel_impacts_sorted) if x["impact"] < 0][:3]
                    
                    total_pos_impact = sum(x["impact"] for x in sel_pos_drivers) + 1e-9
                    sum_displayed_pct = 0.0

                    with sh_col1:
                        st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#1E293B;margin-bottom:8px;'>Primary Factors Elevating Risk Score</div>", unsafe_allow_html=True)
                        for item in sel_top_drivers:
                            fn_tuple = FRIENDLY_NAMES.get(item["feature"], (item["feature"], ""))
                            fname, fdesc = fn_tuple if isinstance(fn_tuple, tuple) else (fn_tuple, "")
                            pct = (item["impact"] / total_pos_impact) * 100
                            sum_displayed_pct += pct
                            bw = min(int(pct * 2), 100)
                            st.markdown(f"""
                            <div style='background:#FFFBFB;border:1px solid #FECACA;border-left:4px solid #EF4444;border-radius:6px;padding:10px 14px;margin-bottom:7px;'>
                                <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                                    <div>
                                        <div style='font-size:0.83rem;font-weight:700;color:#1E293B;'>{fname}</div>
                                        <div style='font-size:0.74rem;color:#64748B;margin-top:2px;'>{fdesc}</div>
                                    </div>
                                    <div style='text-align:right;flex-shrink:0;margin-left:14px;'>
                                        <div style='font-size:0.62rem;color:#94A3B8;text-transform:uppercase;font-weight:600;'>Contribution</div>
                                        <div style='font-size:0.95rem;font-weight:700;color:#B91C1C;'>{pct:.0f}%</div>
                                    </div>
                                </div>
                                <div style='margin-top:6px;background:#FEE2E2;border-radius:3px;height:4px;'>
                                    <div style='background:#EF4444;width:{bw}%;height:4px;border-radius:3px;'></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        remainder_pct = max(0.0, 100.0 - sum_displayed_pct)
                        if remainder_pct >= 0.5:
                            bw_rem = min(int(remainder_pct * 2), 100)
                            st.markdown(f"""
                            <div style='background:#FFFBFB;border:1px solid #FECACA;border-left:4px solid #F97316;border-radius:6px;padding:10px 14px;margin-bottom:7px;'>
                                <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                                    <div>
                                        <div style='font-size:0.83rem;font-weight:700;color:#1E293B;'>Other Contributing Factors</div>
                                        <div style='font-size:0.74rem;color:#64748B;margin-top:2px;'>Cumulative impact of other minor features contributing to elevated risk.</div>
                                    </div>
                                    <div style='text-align:right;flex-shrink:0;margin-left:14px;'>
                                        <div style='font-size:0.62rem;color:#94A3B8;text-transform:uppercase;font-weight:600;'>Contribution</div>
                                        <div style='font-size:0.95rem;font-weight:700;color:#C2410C;'>{remainder_pct:.0f}%</div>
                                    </div>
                                </div>
                                <div style='margin-top:6px;background:#FED7AA;border-radius:3px;height:4px;'>
                                    <div style='background:#F97316;width:{bw_rem}%;height:4px;border-radius:3px;'></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        if sel_top_mitigators:
                            st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#1E293B;margin-top:12px;margin-bottom:8px;'>Mitigating Factors (Consistent with Normal Patterns)</div>", unsafe_allow_html=True)
                            for item in sel_top_mitigators:
                                fn_tuple = FRIENDLY_NAMES.get(item["feature"], (item["feature"], ""))
                                fname, fdesc = fn_tuple if isinstance(fn_tuple, tuple) else (fn_tuple, "")
                                st.markdown(f"""
                                <div style='background:#F8FFFE;border:1px solid #BBF7D0;border-left:4px solid #22C55E;border-radius:6px;padding:8px 12px;margin-bottom:6px;'>
                                    <div style='font-size:0.8rem;font-weight:600;color:#1E293B;'>{fname}</div>
                                    <div style='font-size:0.72rem;color:#64748B;margin-top:1px;'>{fdesc} Consistent with historical norms.</div>
                                </div>
                                """, unsafe_allow_html=True)

                    with sh_col2:
                        st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#1E293B;margin-bottom:8px;'>SHAP Waterfall Attribution Plot</div>", unsafe_allow_html=True)
                        base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                        explanation = shap.Explanation(
                            values=sel_shap_vals,
                            base_values=base_val,
                            data=sel_feat_df.iloc[0].values,
                            feature_names=list(sel_feat_df.columns)
                        )
                        fig, ax = plt.subplots(figsize=(7, 4.5))
                        shap.plots.waterfall(explanation, max_display=8, show=False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
        else:
            st.info("No high-risk transactions were identified in the uploaded ledger based on the current detection threshold.")
    else:
        st.markdown("<div style='background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:60px 32px;text-align:center;margin-top:8px;'><div style='font-size:2.2rem;margin-bottom:10px;opacity:0.25;'>&#128196;</div><div style='font-size:0.98rem;font-weight:700;color:#0F172A;margin-bottom:5px;'>No Data Loaded</div><div style='font-size:0.85rem;color:#64748B;max-width:380px;margin:0 auto;'>Upload a corporate expense ledger CSV above to run the risk assessment engine and populate the dashboard.</div></div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    
    # Load High-Risk Example button
    top_c1, top_c2 = st.columns([3, 1.2])
    with top_c1:
        st.markdown("<div style='background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:16px 22px;margin-bottom:16px;'><div style='font-size:0.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.1em;'>Single Transaction Evaluation</div><div style='font-size:0.98rem;font-weight:700;color:#0F172A;margin-top:2px;'>Assess Transaction Risk</div><div style='font-size:0.82rem;color:#64748B;margin-top:2px;'>Enter transaction parameters to evaluate risk level and generate auditor recommendations.</div></div>", unsafe_allow_html=True)
    with top_c2:
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
        if st.button("Load sample high-risk transaction", help="Prefill the form with a high-risk anomalous transaction"):
            st.session_state["single_cat"] = "Client Meeting"
            st.session_state["single_amt"] = 27500.0
            st.session_state["single_pay"] = "Reimbursement"
            st.session_state["single_dept"] = "Sales"
            st.session_state["single_role"] = "Sales Executive"
            st.session_state["single_hour"] = 2
            st.session_state["single_day"] = 28
            st.session_state["single_emp_avg"] = 2500.0
            st.session_state["single_cat_avg"] = 3160.0
            st.session_state["single_mtd"] = 45000.0
            st.session_state["single_freq"] = 16
            st.rerun()

    # Form defaults from session_state
    def_cat = st.session_state.get("single_cat", encodings["expense_category"]["values"][0])
    def_cat_idx = encodings["expense_category"]["values"].index(def_cat) if def_cat in encodings["expense_category"]["values"] else 0
    def_pay = st.session_state.get("single_pay", encodings["payment_method"]["values"][0])
    def_pay_idx = encodings["payment_method"]["values"].index(def_pay) if def_pay in encodings["payment_method"]["values"] else 0
    def_dept = st.session_state.get("single_dept", encodings["department"]["values"][0])
    def_dept_idx = encodings["department"]["values"].index(def_dept) if def_dept in encodings["department"]["values"] else 0
    def_role = st.session_state.get("single_role", encodings["employee_role"]["values"][0])
    def_role_idx = encodings["employee_role"]["values"].index(def_role) if def_role in encodings["employee_role"]["values"] else 0

    with st.form("txn_risk_form", border=False):
        st.markdown("<div style='background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:22px;'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div style='font-size:0.68rem;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;border-bottom:1px solid #F1F5F9;padding-bottom:8px;'>Transaction Details</div>", unsafe_allow_html=True)
            expense_category = st.selectbox("Expense Category", encodings["expense_category"]["values"], index=def_cat_idx)
            amount = st.number_input("Amount (Rs.)", min_value=0.0, value=float(st.session_state.get("single_amt", 25000.0)), step=100.0)
            payment_method = st.selectbox("Payment Method", encodings["payment_method"]["values"], index=def_pay_idx)
        with col2:
            st.markdown("<div style='font-size:0.68rem;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;border-bottom:1px solid #F1F5F9;padding-bottom:8px;'>Employee Context</div>", unsafe_allow_html=True)
            department = st.selectbox("Department", encodings["department"]["values"], index=def_dept_idx)
            employee_role = st.selectbox("Role", encodings["employee_role"]["values"], index=def_role_idx)
            expense_hour = st.slider("Submission Hour", 0, 23, int(st.session_state.get("single_hour", 14)), help="0=Midnight, 23=11PM")
            expense_day = st.slider("Day of Month", 1, 31, int(st.session_state.get("single_day", 15)))
        with col3:
            st.markdown("<div style='font-size:0.68rem;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;border-bottom:1px solid #F1F5F9;padding-bottom:8px;'>Historical Baselines</div>", unsafe_allow_html=True)
            emp_cat_avg = st.number_input("Employee Historical Average (Rs.)", value=float(st.session_state.get("single_emp_avg", 1500.0)))
            cat_avg = st.number_input("Company Category Benchmark (Rs.)", value=float(st.session_state.get("single_cat_avg", 1200.0)))
            mtd_spend = st.number_input("Month-to-Date Spend (Rs.)", value=float(st.session_state.get("single_mtd", 35000.0)))
            tx_freq = st.number_input("Monthly Transaction Count", value=int(st.session_state.get("single_freq", 5)))
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Assess Transaction Risk", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        # Form Validation
        has_error = False
        if amount <= 0:
            st.error("Validation Error: Transaction Amount must be greater than Rs. 0.")
            has_error = True
        if mtd_spend < amount:
            st.error(f"Validation Error: Month-to-Date Spend (Rs. {mtd_spend:,.2f}) cannot be less than the current transaction amount (Rs. {amount:,.2f}). Month-to-date spend reflects total cumulative monthly spend.")
            has_error = True

        if not has_error:
            row = {
                "amount": amount,
                "expense_category": expense_category,
                "department": department,
                "employee_role": employee_role,
                "payment_method": payment_method,
                "hour": expense_hour,
                "day_of_month": expense_day,
                "is_weekend": 0,
                "employee_category_avg": emp_cat_avg,
                "category_avg_amount": cat_avg,
                "monthly_employee_spend": mtd_spend,
                "transaction_frequency": tx_freq
            }
            with st.spinner("Running risk assessment..."):
                feature_df = engineer_features(row)
                proba, flag, comp = predict(feature_df)
                shap_raw = explainer.shap_values(feature_df)
                shap_vals = shap_raw[1][0] if isinstance(shap_raw, list) else shap_raw[0]
                tier_label, urgency = assign_priority_tier(comp)

            divider()
            
            # Badge styles per tier
            badge_styles = {
                "Tier 1 - Critical": ("#EF4444", "#FEF2F2", "#B91C1C", "HIGH RISK - AUDIT REVIEW REQUIRED"),
                "Tier 2 - High Risk": ("#F97316", "#FFF7ED", "#C2410C", "ELEVATED RISK - EXPEDITED REVIEW RECOMMENDED"),
                "Tier 3 - Moderate": ("#EAB308", "#FEFCE8", "#A16207", "MODERATE RISK - ROUTINE MONITORING"),
                "Tier 4 - Low Risk": ("#22C55E", "#F0FDF4", "#15803D", "LOW RISK - WITHIN ACCEPTABLE PARAMETERS")
            }
            bd_col, bg_col, txt_col, banner_text = badge_styles.get(tier_label, ("#64748B", "#F8FAFC", "#0F172A", "RISK ASSESSMENT COMPLETE"))

            st.markdown(f"""
            <div style='background:{bg_col};border:1px solid {bd_col};border-left:5px solid {bd_col};border-radius:8px;padding:16px 20px;margin-bottom:18px;display:flex;justify-content:space-between;align-items:center;'>
                <div>
                    <div style='font-size:0.68rem;font-weight:700;color:{txt_col};text-transform:uppercase;letter-spacing:0.1em;'>Risk Assessment Result</div>
                    <div style='font-size:1.1rem;font-weight:700;color:{txt_col};margin-top:2px;'>{banner_text}</div>
                </div>
                <div>
                    <span style='background:{bd_col};color:white;padding:6px 16px;border-radius:20px;font-size:0.85rem;font-weight:700;letter-spacing:0.04em;'>{tier_label}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Composite Risk Score",
                f"{comp:.3f}",
                help="Weighted composite score combining model probability (50%), statistical deviation (25%), transaction velocity (15%), and off-hours timing (10%)."
            )
            m2.metric(
                "Model Risk Probability",
                format_probability(proba),
                help="Estimated probability from the tree-based model that the transaction represents an anomalous spending pattern requiring audit attention."
            )
            m3.metric(
                "Risk Tier",
                tier_label,
                help="Classification tier derived from the composite risk score for audit queue routing."
            )
            m4.metric(
                "Statistical Deviation",
                f"{float(feature_df['modified_z_score'].values[0]):.1f} SD",
                help="Modified Z-score: 0.6745 × (Amount − Category Median) / MAD. Measures robust standard deviations from the historical category median spend."
            )

            # Score-to-Tier Mapping Legend
            st.markdown("""
            <div style='background:#FFFFFF;border:1px solid #E2E8F0;border-radius:6px;padding:10px 16px;margin:14px 0;display:flex;flex-wrap:wrap;gap:18px;align-items:center;'>
                <span style='font-size:0.72rem;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;'>Score-to-Tier Mapping Legend:</span>
                <span style='font-size:0.78rem;color:#1E293B;'><strong style='color:#EF4444;'>Tier 1 - Critical</strong> (&ge; 0.85)</span>
                <span style='font-size:0.78rem;color:#1E293B;'><strong style='color:#F97316;'>Tier 2 - High Risk</strong> (&ge; 0.70)</span>
                <span style='font-size:0.78rem;color:#1E293B;'><strong style='color:#EAB308;'>Tier 3 - Moderate</strong> (&ge; 0.55)</span>
                <span style='font-size:0.78rem;color:#1E293B;'><strong style='color:#22C55E;'>Tier 4 - Low Risk</strong> (&lt; 0.55)</span>
            </div>
            """, unsafe_allow_html=True)

            divider()
            rcol, acol = st.columns([3, 2])
            impacts = [{"feature": col, "impact": shap_vals[i], "value": feature_df.iloc[0, i]} for i, col in enumerate(FEATURE_COLUMNS)]
            impacts_sorted = sorted(impacts, key=lambda x: x["impact"], reverse=True)
            pos_drivers = [x for x in impacts_sorted if x["impact"] > 0]
            top_drivers = pos_drivers[:4]
            top_mitigators = [x for x in reversed(impacts_sorted) if x["impact"] < 0][:3]
            
            total_pos_impact = sum(x["impact"] for x in pos_drivers) + 1e-9
            sum_displayed_pct = 0.0

            with rcol:
                if flag and top_drivers:
                    st.markdown("<div style='font-size:0.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;'>Risk Factor Attribution</div><div style='font-size:0.97rem;font-weight:700;color:#0F172A;margin-bottom:3px;'>Why Does This Require Audit Attention?</div><div style='font-size:0.8rem;color:#64748B;margin-bottom:12px;'>The risk model identified the following primary factors contributing to the elevated score.</div>", unsafe_allow_html=True)
                    for item in top_drivers:
                        fn_tuple = FRIENDLY_NAMES.get(item["feature"], (item["feature"], ""))
                        fname, fdesc = fn_tuple if isinstance(fn_tuple, tuple) else (fn_tuple, "")
                        pct = (item["impact"] / total_pos_impact) * 100
                        sum_displayed_pct += pct
                        bw = min(int(pct * 2), 100)
                        st.markdown(f"""
                        <div style='background:#FFFBFB;border:1px solid #FECACA;border-left:4px solid #EF4444;border-radius:6px;padding:12px 14px;margin-bottom:8px;'>
                            <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                                <div>
                                    <div style='font-size:0.85rem;font-weight:700;color:#1E293B;'>{fname}</div>
                                    <div style='font-size:0.75rem;color:#64748B;margin-top:2px;line-height:1.4;'>{fdesc}</div>
                                </div>
                                <div style='text-align:right;flex-shrink:0;margin-left:14px;'>
                                    <div style='font-size:0.65rem;color:#94A3B8;text-transform:uppercase;font-weight:600;'>Contribution</div>
                                    <div style='font-size:0.98rem;font-weight:700;color:#B91C1C;'>{pct:.0f}%</div>
                                </div>
                            </div>
                            <div style='margin-top:8px;background:#FEE2E2;border-radius:3px;height:4px;'>
                                <div style='background:#EF4444;width:{bw}%;height:4px;border-radius:3px;'></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    rem_pct = max(0.0, 100.0 - sum_displayed_pct)
                    if rem_pct >= 0.5:
                        bw_rem = min(int(rem_pct * 2), 100)
                        st.markdown(f"""
                        <div style='background:#FFFBFB;border:1px solid #FECACA;border-left:4px solid #F97316;border-radius:6px;padding:12px 14px;margin-bottom:8px;'>
                            <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                                <div>
                                    <div style='font-size:0.85rem;font-weight:700;color:#1E293B;'>Other Contributing Factors</div>
                                    <div style='font-size:0.75rem;color:#64748B;margin-top:2px;line-height:1.4;'>Cumulative effect of other secondary features contributing to the overall risk score.</div>
                                </div>
                                <div style='text-align:right;flex-shrink:0;margin-left:14px;'>
                                    <div style='font-size:0.65rem;color:#94A3B8;text-transform:uppercase;font-weight:600;'>Contribution</div>
                                    <div style='font-size:0.98rem;font-weight:700;color:#C2410C;'>{rem_pct:.0f}%</div>
                                </div>
                            </div>
                            <div style='margin-top:8px;background:#FED7AA;border-radius:3px;height:4px;'>
                                <div style='background:#F97316;width:{bw_rem}%;height:4px;border-radius:3px;'></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                elif not flag:
                    st.markdown("<div style='background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;padding:18px;'><div style='font-size:0.85rem;font-weight:700;color:#15803D;'>Transaction is within acceptable parameters.</div><div style='font-size:0.8rem;color:#64748B;margin-top:3px;'>No significant risk factors were identified. This transaction does not require immediate audit attention based on the current model configuration.</div></div>", unsafe_allow_html=True)
                
                st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
                if top_mitigators:
                    st.markdown("<div style='font-size:0.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;'>Mitigating Factors</div><div style='font-size:0.9rem;font-weight:700;color:#0F172A;margin-bottom:10px;'>Factors Consistent with Normal Behaviour</div>", unsafe_allow_html=True)
                    for item in top_mitigators:
                        fn_tuple = FRIENDLY_NAMES.get(item["feature"], (item["feature"], ""))
                        fname, fdesc = fn_tuple if isinstance(fn_tuple, tuple) else (fn_tuple, "")
                        st.markdown(f"<div style='background:#F8FFFE;border:1px solid #BBF7D0;border-left:4px solid #22C55E;border-radius:6px;padding:10px 14px;margin-bottom:7px;'><div style='font-size:0.83rem;font-weight:600;color:#1E293B;'>{fname}</div><div style='font-size:0.75rem;color:#64748B;margin-top:2px;'>{fdesc} Consistent with historical norms.</div></div>", unsafe_allow_html=True)

                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                with st.expander("Advanced Explainability - For Data Science / Analytics Team"):
                    st.markdown("<div style='font-size:0.8rem;color:#64748B;margin-bottom:10px;'>SHAP (SHapley Additive exPlanations) values show how individual model features contributed to the calculated risk score. Positive values increase risk; negative values decrease risk.</div>", unsafe_allow_html=True)
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                    explanation = shap.Explanation(values=shap_vals, base_values=base_val, data=feature_df.iloc[0].values, feature_names=list(feature_df.columns))
                    fig, ax = plt.subplots(figsize=(8, 5))
                    shap.plots.waterfall(explanation, max_display=10, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            with acol:
                if flag:
                    at = "Immediate audit review recommended."
                    ac = "#B91C1C"; abg = "#FEF2F2"; abd = "#EF4444"
                    checklist = ["Request supporting documentation and receipts", "Verify business justification for the expense", "Review approval chain and authorisation records", "Cross-reference with employee travel and activity records", "Assess consistency with prior submission patterns"]
                else:
                    at = "No immediate audit action required."
                    ac = "#15803D"; abg = "#F0FDF4"; abd = "#22C55E"
                    checklist = ["Transaction may be retained in routine monitoring queue", "No escalation required at current threshold"]
                items_html = "".join(f"<div style='display:flex;gap:8px;margin-bottom:6px;'><span style='color:{ac};font-size:0.72rem;margin-top:1px;'>&#10003;</span><span style='font-size:0.78rem;color:#475569;'>{item}</span></div>" for item in checklist)
                st.markdown(f"<div style='background:{abg};border:1px solid {abd};border-radius:8px;padding:18px 20px;margin-bottom:14px;'><div style='font-size:0.68rem;font-weight:700;color:{ac};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:5px;'>Recommended Action</div><div style='font-size:0.92rem;font-weight:700;color:#0F172A;margin-bottom:10px;'>{at}</div>{items_html}</div>", unsafe_allow_html=True)
                st.markdown("<div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:13px 15px;'><div style='font-size:0.65rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;'>Important Disclaimer</div><div style='font-size:0.75rem;color:#64748B;line-height:1.5;'>This system prioritizes transactions for audit review based on statistical risk indicators. It does not determine policy violations or misconduct. Final audit decisions remain with the authorised auditor.</div></div>", unsafe_allow_html=True)
