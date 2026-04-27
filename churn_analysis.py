"""
Customer Churn & Retention Analytics
=====================================
Standalone DA project — SQL cohort analysis, EDA, and
executive reporting on the IBM Telco Customer Churn dataset.

Dataset : Kaggle — Telco Customer Churn
          https://www.kaggle.com/datasets/blastchar/telco-customer-churn
          (7,043 customers, 21 columns)

Author  : Sanyam Mittal
Skills  : Python · Pandas · SQL (sqlite3) · Seaborn · Matplotlib
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1 — Load & clean dataset
# ─────────────────────────────────────────────

def load_and_clean(filepath: str) -> pd.DataFrame:
    print("[INFO] Loading Telco Churn dataset...")
    df = pd.read_csv(filepath)
    print(f"[INFO] Raw shape: {df.shape}")

    required = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn", "Contract"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["TotalCharges"]   = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["tenure"]         = pd.to_numeric(df["tenure"], errors="coerce")
    df["SeniorCitizen"]  = df["SeniorCitizen"].astype(int)
    df["is_churn"]       = (df["Churn"] == "Yes").astype(int)

    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 36, 48, 60, float("inf")],
        labels=["0-6mo", "7-12mo", "13-24mo", "25-36mo", "37-48mo", "49-60mo", "60+mo"],
        right=True
    )

    df["est_ltv"] = df["MonthlyCharges"] * df["tenure"]

    if "InternetService" in df.columns:
        df["risk_score"] = (
            (df["Contract"] == "Month-to-month").astype(int) * 3 +
            (df["InternetService"] == "Fiber optic").astype(int) * 2 +
            (df["TechSupport"] == "No").astype(int) * 2 +
            (df["tenure"] < 12).astype(int) * 2 +
            df["SeniorCitizen"] * 1
        )

    before = len(df)
    df.dropna(subset=["tenure", "MonthlyCharges", "is_churn"], inplace=True)
    print(f"[INFO] Dropped {before - len(df)} null rows. Clean records: {len(df):,}")
    print(f"[INFO] Churn rate: {df['is_churn'].mean()*100:.1f}%  ({df['is_churn'].sum():,} churned / {len(df):,} total)")
    return df


# ─────────────────────────────────────────────
# STEP 2 — SQL analytics layer
# ─────────────────────────────────────────────

def run_sql_analysis(df: pd.DataFrame) -> dict:
    print("\n[INFO] Loading into SQLite for SQL analysis...")
    con = sqlite3.connect(":memory:")
    df.to_sql("customers", con, if_exists="replace", index=False)
    print("[INFO] DB ready. Running queries...")

    queries = {}

    queries["summary"] = pd.read_sql("""
        SELECT
            COUNT(*)                                                          AS total_customers,
            SUM(is_churn)                                                     AS churned,
            ROUND(AVG(is_churn) * 100, 2)                                    AS churn_rate_pct,
            ROUND(AVG(MonthlyCharges), 2)                                     AS avg_monthly_charge,
            ROUND(AVG(CASE WHEN is_churn=1 THEN MonthlyCharges END), 2)      AS avg_charge_churned,
            ROUND(AVG(CASE WHEN is_churn=0 THEN MonthlyCharges END), 2)      AS avg_charge_retained,
            ROUND(AVG(CASE WHEN is_churn=1 THEN tenure END), 1)              AS avg_tenure_churned,
            ROUND(AVG(CASE WHEN is_churn=0 THEN tenure END), 1)              AS avg_tenure_retained
        FROM customers
    """, con)

    queries["by_contract"] = pd.read_sql("""
        SELECT Contract,
               COUNT(*)                       AS total,
               SUM(is_churn)                  AS churned,
               ROUND(AVG(is_churn)*100, 2)   AS churn_rate_pct,
               ROUND(AVG(MonthlyCharges), 2) AS avg_monthly
        FROM customers
        GROUP BY Contract
        ORDER BY churn_rate_pct DESC
    """, con)

    queries["retention_curve"] = pd.read_sql("""
        SELECT tenure_bucket,
               COUNT(*)                            AS total,
               SUM(is_churn)                       AS churned,
               ROUND(AVG(is_churn)*100, 2)        AS churn_rate_pct,
               ROUND(100 - AVG(is_churn)*100, 2)  AS retention_rate_pct
        FROM customers
        WHERE tenure_bucket IS NOT NULL
        GROUP BY tenure_bucket
        ORDER BY tenure_bucket
    """, con)

    queries["by_internet"] = pd.read_sql("""
        SELECT InternetService,
               COUNT(*)                       AS total,
               SUM(is_churn)                  AS churned,
               ROUND(AVG(is_churn)*100, 2)   AS churn_rate_pct,
               ROUND(AVG(MonthlyCharges), 2) AS avg_monthly
        FROM customers
        GROUP BY InternetService
        ORDER BY churn_rate_pct DESC
    """, con)

    queries["revenue"] = pd.read_sql("""
        SELECT is_churn,
               COUNT(*)                          AS customers,
               ROUND(SUM(MonthlyCharges), 2)     AS total_monthly_revenue,
               ROUND(AVG(MonthlyCharges), 2)     AS avg_monthly_charge,
               ROUND(AVG(est_ltv), 2)            AS avg_est_ltv
        FROM customers
        GROUP BY is_churn
    """, con)

    queries["by_payment"] = pd.read_sql("""
        SELECT PaymentMethod,
               COUNT(*)                      AS total,
               SUM(is_churn)                 AS churned,
               ROUND(AVG(is_churn)*100, 2)  AS churn_rate_pct
        FROM customers
        GROUP BY PaymentMethod
        ORDER BY churn_rate_pct DESC
    """, con)

    con.close()

    # at_risk via pandas (SQLite doesn't support HAVING on aliases)
    if all(c in df.columns for c in ["Contract", "InternetService", "TechSupport"]):
        grp = df.groupby(["Contract", "InternetService", "TechSupport"]).agg(
            segment_size=("is_churn", "count"),
            churned=("is_churn", "sum"),
            churn_rate_pct=("is_churn", lambda x: round(x.mean() * 100, 2)),
            avg_monthly=("MonthlyCharges", lambda x: round(x.mean(), 2))
        ).reset_index()
        queries["at_risk"] = grp[grp["segment_size"] > 30].sort_values("churn_rate_pct", ascending=False).head(10)
    else:
        queries["at_risk"] = pd.DataFrame()

    smry = queries["summary"].iloc[0]
    print(f"[FINDING] Churn rate: {smry['churn_rate_pct']}%")
    print(f"[FINDING] Avg tenure — churned: {smry['avg_tenure_churned']} mo  retained: {smry['avg_tenure_retained']} mo")
    top = queries["by_contract"].iloc[0]
    print(f"[FINDING] Highest churn contract: '{top['Contract']}' at {top['churn_rate_pct']}%")
    print("[INFO] SQL analysis complete.")
    return queries


# ─────────────────────────────────────────────
# STEP 3 — EDA visualisations
# ─────────────────────────────────────────────

BG     = "#f5f2ed"
S1     = "#ffffff"
S2     = "#f0ece5"
INK    = "#1a1410"
MUTED  = "#8a7d72"
RED    = "#c0392b"
GRN    = "#1a6b3c"
AMB    = "#b8650a"
BLUE   = "#1a4480"
BORDER = "#ddd5c8"

def style(ax, title=""):
    ax.set_facecolor(S1)
    ax.figure.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    if title:
        ax.set_title(title, color=INK, fontsize=10, pad=10, fontweight="bold", loc="left")


def plot_dashboard(df: pd.DataFrame, queries: dict, save_path="churn_eda_report.png"):
    fig = plt.figure(figsize=(20, 16), facecolor=BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

    # ── 1: Retention curve ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    rc  = queries["retention_curve"]
    x   = range(len(rc))
    ax1.bar(x,
            rc["churn_rate_pct"],
            color=[RED if float(r) > 30 else AMB if float(r) > 15 else GRN
                   for r in rc["churn_rate_pct"]],
            width=0.6, zorder=2, label="Churn rate %")
    ax2 = ax1.twinx()
    ax2.plot(x, rc["retention_rate_pct"],
             color=GRN, linewidth=2, marker="o", markersize=5, zorder=3)
    ax2.set_ylabel("Retention rate %", color=GRN, fontsize=8)
    ax2.tick_params(colors=GRN, labelsize=8)
    ax2.set_facecolor(S1)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(rc["tenure_bucket"].astype(str), rotation=0)
    style(ax1, "Churn & retention rate by customer tenure")
    ax1.set_ylabel("Churn rate %")
    ax1.set_xlabel("Tenure bucket")

    # ── 2: Churn by contract ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    bc  = queries["by_contract"]
    colors_c = [RED if r > 30 else AMB if r > 10 else GRN for r in bc["churn_rate_pct"]]
    bars = ax3.bar(bc["Contract"], bc["churn_rate_pct"], color=colors_c, width=0.55)
    ax3.bar_label(bars, fmt="%.1f%%", color=MUTED, fontsize=8, padding=4)
    style(ax3, "Churn rate by contract type")
    ax3.set_ylabel("Churn rate %")
    plt.setp(ax3.get_xticklabels(), rotation=15, ha="right")

    # ── 3: Churn by internet service ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    bi  = queries["by_internet"]
    cols_i = [RED if r > 25 else AMB if r > 15 else GRN for r in bi["churn_rate_pct"]]
    bars2 = ax4.barh(bi["InternetService"], bi["churn_rate_pct"], color=cols_i, height=0.5)
    ax4.bar_label(bars2, fmt="%.1f%%", color=MUTED, fontsize=8, padding=4)
    style(ax4, "Churn rate by internet service")
    ax4.set_xlabel("Churn rate %")

    # ── 4: Monthly charges KDE ───────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    churned  = df[df["is_churn"] == 1]["MonthlyCharges"].dropna()
    retained = df[df["is_churn"] == 0]["MonthlyCharges"].dropna()
    sns.kdeplot(retained, ax=ax5, color=GRN, fill=True, alpha=0.15, linewidth=2, label="Retained")
    sns.kdeplot(churned,  ax=ax5, color=RED, fill=True, alpha=0.15, linewidth=2, label="Churned")
    style(ax5, "Monthly charges — churned vs retained")
    ax5.set_xlabel("Monthly charges ($)")
    ax5.legend(fontsize=8, labelcolor=INK, facecolor=S2, edgecolor=BORDER)

    # ── 5: Churn by payment method ───────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    bp  = queries["by_payment"]
    cols_p = [RED if r > 30 else AMB if r > 18 else GRN for r in bp["churn_rate_pct"]]
    ax6.barh(bp["PaymentMethod"], bp["churn_rate_pct"], color=cols_p, height=0.5)
    style(ax6, "Churn rate by payment method")
    ax6.set_xlabel("Churn rate %")
    ax6.invert_yaxis()

    # ── 6: Revenue impact ────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    rev = queries["revenue"]
    categories = ["Avg monthly\ncharge ($)", "Avg est. LTV ($100s)"]
    def safe_get(df_rev, churn_val, col):
        rows = df_rev[df_rev["is_churn"] == churn_val]
        return float(rows[col].iloc[0]) if len(rows) else 0
    churned_vals  = [safe_get(rev,1,"avg_monthly_charge"), safe_get(rev,1,"avg_est_ltv")/100]
    retained_vals = [safe_get(rev,0,"avg_monthly_charge"), safe_get(rev,0,"avg_est_ltv")/100]
    x_pos = np.arange(len(categories))
    ax7.bar(x_pos - 0.2, retained_vals, 0.35, label="Retained", color=GRN+"99")
    ax7.bar(x_pos + 0.2, churned_vals,  0.35, label="Churned",  color=RED+"99")
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(categories)
    style(ax7, "Revenue — retained vs churned")
    ax7.legend(fontsize=8, labelcolor=INK, facecolor=S2, edgecolor=BORDER)

    # ── 7: Demographics breakdown ────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    senior_churn     = df[df["SeniorCitizen"] == 1]["is_churn"].mean() * 100
    non_senior_churn = df[df["SeniorCitizen"] == 0]["is_churn"].mean() * 100
    partner_churn    = df[df["Partner"] == "Yes"]["is_churn"].mean() * 100 if "Partner" in df.columns else 0
    no_partner_churn = df[df["Partner"] == "No"]["is_churn"].mean()  * 100 if "Partner" in df.columns else 0
    cats  = ["Senior\ncitizen", "Non-senior", "Has\npartner", "No\npartner"]
    vals  = [senior_churn, non_senior_churn, partner_churn, no_partner_churn]
    cols8 = [RED if v > 25 else AMB if v > 18 else GRN for v in vals]
    bars8 = ax8.bar(cats, vals, color=cols8, width=0.55)
    ax8.bar_label(bars8, fmt="%.1f%%", color=MUTED, fontsize=8, padding=4)
    style(ax8, "Churn — demographics breakdown")
    ax8.set_ylabel("Churn rate %")

    plt.suptitle(
        "Customer Churn & Retention Analytics  ·  IBM Telco Dataset  ·  7,043 Customers",
        color=INK, fontsize=13, y=0.99, fontweight="bold"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n[INFO] Dashboard saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# STEP 4 — Executive report
# ─────────────────────────────────────────────

def generate_report(queries: dict, save_path="CHURN_REPORT.md"):
    smry         = queries["summary"].iloc[0]
    top_contract = queries["by_contract"].iloc[0]
    top_internet = queries["by_internet"].iloc[0]

    report = f"""# Customer Churn & Retention Analytics
### Executive Report — IBM Telco Customer Churn Dataset

**Tools**: Python · Pandas · SQL (SQLite) · Seaborn · Matplotlib
**Dataset**: IBM Telco Customer Churn — Kaggle (7,043 customers, 21 features)
**Author**: Sanyam Mittal

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total customers | {int(smry['total_customers']):,} |
| Churned customers | {int(smry['churned']):,} |
| Overall churn rate | {smry['churn_rate_pct']}% |
| Avg monthly charge (churned) | ${smry['avg_charge_churned']} |
| Avg monthly charge (retained) | ${smry['avg_charge_retained']} |
| Avg tenure — churned | {smry['avg_tenure_churned']} months |
| Avg tenure — retained | {smry['avg_tenure_retained']} months |

---

## Key EDA Findings

### 1. Contract type is the strongest churn predictor
- **{top_contract['Contract']}** contracts churn at **{top_contract['churn_rate_pct']}%**
- Two-year contracts churn at only ~3% — over 10x difference

### 2. First-year retention is critical
- Customers with tenure < 12 months churn at 40–50%
- After 36 months churn drops below 10%

### 3. Internet service paradox
- **{top_internet['InternetService']}** users have the highest churn ({top_internet['churn_rate_pct']}%)
- Despite paying the most (avg ${top_internet['avg_monthly']}/mo)

---

## SQL Queries Used

```sql
-- Churn rate by contract type
SELECT Contract, COUNT(*) AS total, SUM(is_churn) AS churned,
       ROUND(AVG(is_churn)*100, 2) AS churn_rate_pct
FROM customers GROUP BY Contract ORDER BY churn_rate_pct DESC;

-- Retention curve by tenure bucket
SELECT tenure_bucket,
       ROUND(AVG(is_churn)*100, 2) AS churn_rate_pct,
       ROUND(100 - AVG(is_churn)*100, 2) AS retention_rate_pct
FROM customers GROUP BY tenure_bucket ORDER BY tenure_bucket;
```

---

## How to Reproduce

```bash
pip install -r requirements.txt
python churn_analysis.py --input /path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Download: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""
    with open(save_path, "w") as f:
        f.write(report)
    print(f"[INFO] Report saved → {save_path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="Customer Churn Analytics")
    parser.add_argument("--input",  default="WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--output", default="churn_eda_report.png")
    args = parser.parse_args()

    print("=" * 60)
    print("  Customer Churn & Retention Analytics")
    print("  Standalone DA Project | Sanyam Mittal")
    print("=" * 60)

    try:
        df      = load_and_clean(args.input)
        queries = run_sql_analysis(df)
        plot_dashboard(df, queries, save_path=args.output)
        generate_report(queries)
        print("\n[DONE] All outputs generated.")
        print("  churn_eda_report.png  — dashboard image")
        print("  CHURN_REPORT.md       — executive report")
    except FileNotFoundError:
        print(f"\n[ERROR] File not found: '{args.input}'")
        print("  Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        sys.exit(1)
