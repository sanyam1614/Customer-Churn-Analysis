# Customer Churn & Retention Analytics

> SQL cohort analysis + EDA dashboard on 7,043 telecom customers.  
> Identifies at-risk segments, retention patterns, and revenue impact.

---

## Dashboard preview

![Customer Churn EDA Dashboard](churn_eda_report.png)

---

## Key findings

| Finding | Value |
|---------|-------|
| Overall churn rate | **26.5%** |
| Month-to-month churn rate | **42.7%** — 15× higher than two-year contracts (2.8%) |
| Revenue at risk (top 3 segments) | **~$996K / month** |
| Highest risk tenure window | **0–12 months** — churn rate 47–52% |
| Churn drops below | **6%** after 36 months of tenure |

---

## Project structure

```
CustomerChurnAnalytics/
├── churn_analysis.py        ← main script: load → SQL → EDA → report
├── churn_eda_report.png     ← 7-chart dashboard output
├── CHURN_REPORT.md          ← auto-generated executive summary
├── requirements.txt         ← dependencies
└── README.md
```

---

## Tools used

| Category | Tools |
|----------|-------|
| Language | Python 3.9+ |
| Data wrangling | Pandas, NumPy |
| SQL analytics | SQLite3 (8 queries) |
| Visualisation | Seaborn, Matplotlib |
| Reporting | Markdown |

---

## Dataset

**IBM Telco Customer Churn** — publicly available on Kaggle  
7,043 customers · 21 features · no signup required

Download: https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
File name after download: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

---

## How to run

```bash
# 1. Clone this repo
git clone https://github.com/sanyammittal/CustomerChurnAnalytics
cd CustomerChurnAnalytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis
python churn_analysis.py --input /path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv

# Outputs produced:
#   churn_eda_report.png   ← dashboard image
#   CHURN_REPORT.md        ← executive report
```

---

## SQL queries used

```sql
-- Churn rate by contract type
SELECT Contract, COUNT(*) AS total, SUM(is_churn) AS churned,
       ROUND(AVG(is_churn) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- Retention curve by tenure bucket
SELECT tenure_bucket,
       ROUND(AVG(is_churn) * 100, 2)        AS churn_rate_pct,
       ROUND(100 - AVG(is_churn) * 100, 2)  AS retention_rate_pct
FROM customers
WHERE tenure_bucket IS NOT NULL
GROUP BY tenure_bucket
ORDER BY tenure_bucket;

-- Cohort: churn rate by contract × tenure
SELECT Contract, tenure_bucket,
       COUNT(*) AS total,
       ROUND(AVG(is_churn) * 100, 2) AS churn_rate_pct
FROM customers
WHERE tenure_bucket IS NOT NULL
GROUP BY Contract, tenure_bucket
ORDER BY Contract, tenure_bucket;

-- Revenue impact: churned vs retained
SELECT is_churn,
       ROUND(SUM(MonthlyCharges), 2)  AS total_monthly_revenue,
       ROUND(AVG(MonthlyCharges), 2)  AS avg_monthly_charge,
       ROUND(AVG(est_ltv), 2)         AS avg_est_ltv
FROM customers
GROUP BY is_churn;
```

---

## Charts in the dashboard

1. **Retention curve** — churn % + retention % by tenure bucket (bar + line)
2. **Churn by contract type** — month-to-month vs one-year vs two-year
3. **Churn by internet service** — fiber optic vs DSL vs no internet
4. **Monthly charges distribution** — KDE plot comparing churned vs retained
5. **Churn by payment method** — electronic check vs others
6. **Revenue impact** — avg monthly charge and LTV side-by-side
7. **Demographics breakdown** — senior citizen and partner status churn rates

---

## Author

**Sanyam Mittal**  
B.Tech Computer Science · VIT Vellore · May 2026  
[LinkedIn](https://www.linkedin.com/in/sanyam-mittal) · [GitHub](https://github.com/sanyammittal)
