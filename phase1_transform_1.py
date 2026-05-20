"""
CSM Portfolio Project — Phase 1
Transform IBM Telco Churn dataset into a CSM book of business
Output: csm_book_of_business.csv
"""

import pandas as pd
import numpy as np

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_excel("Telco_customer_churn.xlsx")
print(f"Loaded {len(df):,} records")

# ── Clean Total Charges ───────────────────────────────────────────────────────
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0)

# ── Rename columns to CSM language ───────────────────────────────────────────
df = df.rename(columns={
    "CustomerID":        "account_id",
    "City":              "city",
    "State":             "state",
    "Tenure Months":     "months_active",
    "Monthly Charges":   "mrr",
    "Total Charges":     "lifetime_revenue",
    "Contract":          "contract_type",
    "Payment Method":    "payment_method",
    "Paperless Billing": "paperless_billing",
    "Tech Support":      "tech_support",
    "Internet Service":  "internet_service",
    "Online Security":   "online_security",
    "Churn Label":       "churned",
    "Churn Score":       "churn_risk_score",
    "Churn Reason":      "churn_reason",
    "CLTV":              "cltv",
    "Gender":            "gender",
    "Senior Citizen":    "senior_citizen",
    "Phone Service":     "phone_service",
    "Multiple Lines":    "multiple_lines",
    "Streaming TV":      "streaming_tv",
    "Streaming Movies":  "streaming_movies",
    "Online Backup":     "online_backup",
    "Device Protection": "device_protection",
    "Partner":           "partner",
    "Dependents":        "dependents",
})

# ── Map contract type to SaaS plan tier ──────────────────────────────────────
tier_map = {
    "Month-to-month": "Starter",
    "One year":        "Growth",
    "Two year":        "Enterprise",
}
df["plan_tier"] = df["contract_type"].map(tier_map)

# ── Map payment method to billing stability ───────────────────────────────────
# Automatic payments = stable, manual = lower stability
billing_stability_map = {
    "Bank transfer (automatic)": "Stable",
    "Credit card (automatic)":   "Stable",
    "Electronic check":          "Manual",
    "Mailed check":              "Manual",
}
df["billing_stability"] = df["payment_method"].map(billing_stability_map)

# ── Derive renewal window from tenure + contract ──────────────────────────────
contract_length_months = {
    "Month-to-month": 1,
    "One year":        12,
    "Two year":        24,
}
df["contract_length_months"] = df["contract_type"].map(contract_length_months)

# Months until next renewal (how far into the current contract period they are)
df["months_until_renewal"] = df["contract_length_months"] - (
    df["months_active"] % df["contract_length_months"].replace(0, 1)
)
# Renewal urgency bucket
def renewal_bucket(months):
    if months <= 30:
        return "30 days"
    elif months <= 60:
        return "60 days"
    elif months <= 90:
        return "90 days"
    else:
        return "90+ days"

df["renewal_window"] = df["months_until_renewal"].apply(renewal_bucket)

# ── Engagement signals ────────────────────────────────────────────────────────
# Tech support usage as engagement proxy
df["support_engaged"] = df["tech_support"].apply(
    lambda x: True if x == "Yes" else False
)

# Product breadth — how many services is the account using
service_cols = ["phone_service", "online_security", "online_backup",
                "device_protection", "streaming_tv", "streaming_movies"]
df["services_count"] = df[service_cols].apply(
    lambda row: sum(1 for v in row if v == "Yes"), axis=1
)

# ── Health score (0–100, weighted) ───────────────────────────────────────────
"""
Signal weights (documented for methodology writeup):
  Contract type    25 pts  — longer contract = stronger commitment signal
  Tenure           25 pts  — longer relationship = lower churn likelihood
  MRR              20 pts  — higher MRR = more invested, more to lose
  Tech support     15 pts  — engaged accounts use support; dark accounts don't
  Billing method   10 pts  — automatic payment = lower involuntary churn risk
  Services count    5 pts  — more products = higher switching cost
"""

def score_contract(contract):
    return {"Month-to-month": 0, "One year": 15, "Two year": 25}.get(contract, 0)

def score_tenure(months):
    # Scale 0–72 months to 0–25 pts
    return round(min(months / 72, 1) * 25)

def score_mrr(mrr):
    # Scale $18–$119 range to 0–20 pts
    return round(min((mrr - 18) / (119 - 18), 1) * 20)

def score_support(support):
    return 15 if support == "Yes" else 0

def score_billing(method):
    return 10 if method in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0

def score_services(count):
    # 0–6 services → 0–5 pts
    return round(min(count / 6, 1) * 5)

df["health_score"] = (
    df["contract_type"].apply(score_contract) +
    df["months_active"].apply(score_tenure) +
    df["mrr"].apply(score_mrr) +
    df["tech_support"].apply(score_support) +
    df["payment_method"].apply(score_billing) +
    df["services_count"].apply(score_services)
)

# ── Risk status from health score ─────────────────────────────────────────────
def risk_status(score):
    if score < 40:
        return "Needs Attention"
    elif score < 70:
        return "On Track"
    else:
        return "Expansion Ready"

df["risk_status"] = df["health_score"].apply(risk_status)

# ── Churn risk label from existing Churn Score ───────────────────────────────
# Churn Score 0–100 where higher = more likely to churn
def churn_risk_label(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"

df["churn_risk_label"] = df["churn_risk_score"].apply(churn_risk_label)

# ── Recommended CS play ───────────────────────────────────────────────────────
def recommended_play(row):
    if row["churned"] == "Yes":
        return "Save Play"
    if row["churn_risk_score"] >= 70 and row["mrr"] >= 70:
        return "Executive Outreach"
    if row["churn_risk_score"] >= 70:
        return "Re-engage"
    if row["months_until_renewal"] <= 30:
        return "QBR Due"
    if row["health_score"] >= 70 and row["services_count"] <= 3:
        return "Upsell Ready"
    return "Monitor"

df["recommended_play"] = df.apply(recommended_play, axis=1)

# ── MRR at risk flag ──────────────────────────────────────────────────────────
df["mrr_at_risk"] = df.apply(
    lambda r: r["mrr"] if r["churn_risk_label"] == "High" else 0, axis=1
)

# ── Select and order final columns ───────────────────────────────────────────
output_cols = [
    "account_id",
    "city",
    "state",
    "plan_tier",
    "months_active",
    "mrr",
    "lifetime_revenue",
    "cltv",
    "contract_type",
    "months_until_renewal",
    "renewal_window",
    "payment_method",
    "billing_stability",
    "tech_support",
    "support_engaged",
    "services_count",
    "internet_service",
    "health_score",
    "risk_status",
    "churn_risk_score",
    "churn_risk_label",
    "mrr_at_risk",
    "recommended_play",
    "churned",
    "churn_reason",
    "paperless_billing",
    "senior_citizen",
    "gender",
]

output = df[output_cols].copy()

# ── Export ────────────────────────────────────────────────────────────────────
output.to_csv("csm_book_of_business.csv", index=False)
print(f"Exported {len(output):,} accounts to csm_book_of_business.csv")

# ── Summary stats ─────────────────────────────────────────────────────────────
print("\n── Book of business summary ──")
print(f"Total MRR:          ${output['mrr'].sum():,.0f}")
print(f"MRR at risk:        ${output['mrr_at_risk'].sum():,.0f}")
print(f"Avg health score:   {output['health_score'].mean():.1f}")
print(f"\nPlan tier breakdown:")
print(output["plan_tier"].value_counts().to_string())
print(f"\nRisk status breakdown:")
print(output["risk_status"].value_counts().to_string())
print(f"\nRecommended play breakdown:")
print(output["recommended_play"].value_counts().to_string())
print(f"\nRenewal window breakdown:")
print(output["renewal_window"].value_counts().to_string())
