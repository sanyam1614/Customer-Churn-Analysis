# Customer Churn & Retention Analytics
### Executive Report — IBM Telco Customer Churn Dataset

**Tools**: Python · Pandas · SQL (SQLite) · Seaborn · Matplotlib
**Dataset**: IBM Telco Customer Churn — Kaggle (7,043 customers, 21 features)
**Author**: Sanyam Mittal

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total customers | 7,043 |
| Churned customers | 1,869 |
| Overall churn rate | 26.54% |
| Avg monthly charge (churned) | $74.44 |
| Avg monthly charge (retained) | $61.27 |
| Avg tenure — churned | 18.0 months |
| Avg tenure — retained | 37.6 months |

---

## Key EDA Findings

### 1. Contract type is the strongest churn predictor
- **Month-to-month** contracts churn at **42.71%**
- Two-year contracts churn at only ~3% — over 10x difference

### 2. First-year retention is critical
- Customers with tenure < 12 months churn at 40–50%
- After 36 months churn drops below 10%

### 3. Internet service paradox
- **Fiber optic** users have the highest churn (41.89%)
- Despite paying the most (avg $91.5/mo)

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
