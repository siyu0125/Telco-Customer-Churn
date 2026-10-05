# IBM Telco Customer Churn: Study Report

Predicting customer **churn** (a customer leaving within the period) from account, service, and billing
attributes. Dataset: IBM sample "Telco Customer Churn" (Kaggle `blastchar/telco-customer-churn`).
Primary metric: **ROC-AUC** (with PR-AUC and F1, given the 26.5% churn base rate).

---

## Executive summary
- **Best model: soft-voting ensemble — ROC-AUC = 0.849** (PR-AUC 0.666, best-F1 0.635).
  Logistic regression is essentially tied (0.848), so the signal is largely linear.
- **Dominant drivers:** contract type, tenure, and billing dynamics. Month-to-month contracts, fiber-optic
  internet, and electronic-check payment mark a high-churn segment; add-on services and long tenure protect.
- **Engineered billing features top importance:** `monthly_vs_avg`, `charges_per_tenure`, and
  `avg_monthly_billed` outrank most raw fields — recent price relative to a customer's historical average
  is the strongest learned signal.
- **Data-quality fix:** `TotalCharges` ships 11 blank strings, all tenure-0 new customers → set to 0.

Pipeline: `src/eda.py` → `src/pipeline.py` (clean+features) → `src/model.py`. A formal write-up is in
[paper.md](paper.md); companion outputs: [phase1_eda.md](phase1_eda.md), [phase4_model.md](phase4_model.md),
[metrics.json](metrics.json).

## Phase 1 — Data Analysis
- **Shape:** 7,043 customers × 21 columns; target `Churn` = 26.5% Yes (imbalanced).
- **Data quality:** `TotalCharges` is stored as text with 11 blank values, all with `tenure == 0`
  (brand-new customers, not yet billed). `SeniorCitizen` is 0/1; nine service columns use
  "No internet/phone service" as a distinct third level.
- **Categorical drivers (Cramér's V vs churn):** Contract 0.41, OnlineSecurity 0.35, TechSupport 0.34,
  InternetService 0.32, PaymentMethod 0.30 lead; `gender` (0.008) and `PhoneService` (0.011) are irrelevant.
- **Numeric (Mann–Whitney, churn vs stay):** tenure 18 vs 38 months, MonthlyCharges \$74 vs \$61,
  TotalCharges \$1.5k vs \$2.6k (all p ≪ 0.001). Churners are newer, pay more per month, less in total.
- Figures: churn distribution, churn-by-category, numeric-by-churn, churn-by-tenure, association ranking.

## Phase 2 — Cleaning (`pipeline.clean`)
- Coerce `TotalCharges` to numeric; impute the 11 tenure-0 blanks with 0.
- Normalize string dtypes (source columns are Arrow-backed strings); keep the "No service" level as signal.

## Phase 3 — Feature Engineering (`pipeline.add_features`)
- `num_services` (count of add-ons), `has_internet`, `streaming_count`, `is_month_to_month`, `auto_payment`.
- Billing dynamics: `charges_per_tenure`, `monthly_x_tenure`, `avg_monthly_billed`,
  `monthly_vs_avg` (current vs historical average), `is_new_customer`.
- One-hot encoding of remaining categoricals → 55 numeric features.

## Phase 4 — Modeling (`model.py`)
- **Design:** StratifiedKFold(5) × 3 seeds; class imbalance handled per model (`class_weight`/`scale_pos_weight`/
  `auto_class_weights`). Logistic-regression baseline + LightGBM + XGBoost + CatBoost + soft-voting ensemble.

  | model | ROC-AUC | PR-AUC | best-F1 |
  |---|---|---|---|
  | Logistic Regression | 0.848 | 0.665 | 0.637 |
  | LightGBM | 0.844 | 0.656 | 0.629 |
  | XGBoost | 0.847 | 0.662 | 0.631 |
  | CatBoost | 0.848 | 0.666 | 0.633 |
  | **Ensemble** | **0.849** | 0.666 | 0.635 |

- **Top features:** `monthly_vs_avg`, `MonthlyCharges`, `TotalCharges`, `charges_per_tenure`,
  `avg_monthly_billed`, `tenure`, then `num_services`, electronic-check, and no-online-security/tech-support.
- The near-tie with logistic regression confirms the churn signal is mostly linear/additive in these
  features; gradient boosting adds little here.

## Limitations & next steps
1. **Signal ceiling** — ROC-AUC ~0.85 is typical for this dataset; the label is a single snapshot with no
   behavioral/temporal history.
2. **Actionability** — for retention, PR-AUC and recall at a business-chosen threshold matter more than AUC;
   the F1-optimal threshold is reported, but cost-sensitive thresholding should follow the retention economics.
3. **No temporal validation** — a time-based split would better estimate deployment performance.

## Reproducibility
```bash
cd telco-churn
../.venv/bin/python src/eda.py       # Phase 1 -> figures + phase1_eda.md
../.venv/bin/python src/model.py     # Phase 4 -> metrics.json + ROC/PR + importance figures
../.venv/bin/python src/build_paper_html.py   # paper.md -> paper.html
```
Seeds {42, 2024, 7}; all metrics are 5-fold × 3-seed out-of-fold.
