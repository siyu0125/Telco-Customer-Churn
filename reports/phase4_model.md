# Phase 4 — Modeling results

- 7043 customers; 55 features; 5-fold x 3 seeds. Primary metric ROC-AUC; PR-AUC and best-F1 reported (churn base rate 26.5%).


| model | ROC-AUC | PR-AUC | best F1 |
|---|---|---|---|
| logreg | 0.8480 | 0.6651 | 0.637 |
| lgb | 0.8439 | 0.6556 | 0.629 |
| xgb | 0.8470 | 0.6624 | 0.631 |
| cat | 0.8483 | 0.6663 | 0.633 |
| **ensemble** | **0.8488** | 0.6656 | 0.635 |

**Best by ROC-AUC: `ensemble` = 0.8488.**


## OOF confusion matrix (ensemble @ F1-optimal thr=0.526)
```
[[4204  970]
 [ 549 1320]]
```

```
              precision    recall  f1-score   support

        Stay       0.88      0.81      0.85      5174
       Churn       0.58      0.71      0.63      1869

    accuracy                           0.78      7043
   macro avg       0.73      0.76      0.74      7043
weighted avg       0.80      0.78      0.79      7043

```


## Top 20 features (LightGBM)
```
monthly_vs_avg                    2053.9
MonthlyCharges                    1537.6
TotalCharges                      1366.0
charges_per_tenure                1351.5
avg_monthly_billed                1297.2
tenure                            1249.6
monthly_x_tenure                  1124.9
num_services                       235.3
PaymentMethod_Electronic check     223.6
PaperlessBilling_No                176.3
OnlineSecurity_No                  171.1
TechSupport_No                     156.9
MultipleLines_No                   154.8
SeniorCitizen                      150.5
OnlineBackup_No                    143.6
streaming_count                    141.5
gender_Female                      139.1
is_month_to_month                  111.7
Contract_One year                   92.9
Dependents_No                       91.7
```
