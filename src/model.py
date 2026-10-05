"""Phase 4 — Modeling for Telco churn.

Logistic-regression baseline + LightGBM/XGBoost/CatBoost, StratifiedKFold x seeds.
Primary metric ROC-AUC (plus PR-AUC and F1 at the F1-optimal threshold, given 26.5% imbalance).
Writes reports/phase4_model.md and reports/metrics.json.

Run from telco-churn/: ../.venv/bin/python src/model.py
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_recall_curve, confusion_matrix, classification_report)
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

import pipeline as P

warnings.filterwarnings("ignore")
SEEDS = [42, 2024, 7]
NS = 5


def fit_logreg(Xtr, ytr, Xva, seed):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced", random_state=seed)
    m.fit(sc.transform(Xtr), ytr)
    return lambda X: m.predict_proba(sc.transform(X))[:, 1]


def fit_lgb(Xtr, ytr, Xva, seed):
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=31, max_depth=6,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                           reg_alpha=1.0, reg_lambda=5.0, min_child_samples=40,
                           random_state=seed, n_jobs=-1, verbose=-1)
    m.fit(Xtr, ytr)
    return lambda X: m.predict_proba(X)[:, 1], m


def fit_xgb(Xtr, ytr, Xva, seed):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    m = xgb.XGBClassifier(n_estimators=500, learning_rate=0.02, max_depth=4, subsample=0.8,
                          colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0, min_child_weight=5,
                          scale_pos_weight=spw, random_state=seed, n_jobs=-1,
                          tree_method="hist", eval_metric="auc")
    m.fit(Xtr, ytr)
    return lambda X: m.predict_proba(X)[:, 1]


def fit_cat(Xtr, ytr, Xva, seed):
    m = CatBoostClassifier(iterations=600, learning_rate=0.02, depth=5, l2_leaf_reg=5.0,
                           auto_class_weights="Balanced", random_seed=seed, verbose=0,
                           allow_writing_files=False)
    m.fit(Xtr, ytr)
    return lambda X: m.predict_proba(X)[:, 1]


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    i = np.nanargmax(f1[:-1])
    return thr[i], f1[i]


def run():
    X, y, feats = P.build()
    Xv = X.values
    n = len(y)
    oof = {k: np.zeros(n) for k in ("logreg", "lgb", "xgb", "cat")}
    importances = np.zeros(len(feats))

    for seed in SEEDS:
        skf = StratifiedKFold(NS, shuffle=True, random_state=seed)
        for tr, va in skf.split(Xv, y):
            oof["logreg"][va] += fit_logreg(Xv[tr], y[tr], Xv[va], seed)(Xv[va]) / len(SEEDS)
            pred_lgb, m_lgb = fit_lgb(Xv[tr], y[tr], Xv[va], seed)
            oof["lgb"][va] += pred_lgb(Xv[va]) / len(SEEDS)
            importances += m_lgb.feature_importances_ / (len(SEEDS) * NS)
            oof["xgb"][va] += fit_xgb(Xv[tr], y[tr], Xv[va], seed)(Xv[va]) / len(SEEDS)
            oof["cat"][va] += fit_cat(Xv[tr], y[tr], Xv[va], seed)(Xv[va]) / len(SEEDS)

    md = ["# Phase 4 — Modeling results\n",
          f"- {n} customers; {len(feats)} features; {NS}-fold x {len(SEEDS)} seeds. "
          "Primary metric ROC-AUC; PR-AUC and best-F1 reported (churn base rate 26.5%).\n",
          "\n| model | ROC-AUC | PR-AUC | best F1 |", "|---|---|---|---|"]
    print(f"{n} rows, {len(feats)} feats\n")
    print(f"{'model':>8} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>6}")
    results = {}
    for k in oof:
        auc = roc_auc_score(y, oof[k]); ap = average_precision_score(y, oof[k])
        thr, f1 = best_f1_threshold(y, oof[k])
        results[k] = {"roc_auc": auc, "pr_auc": ap, "f1": f1, "thr": float(thr)}
        print(f"{k:>8} {auc:8.4f} {ap:8.4f} {f1:6.3f}")
        md.append(f"| {k} | {auc:.4f} | {ap:.4f} | {f1:.3f} |")

    ens = np.mean([oof[k] for k in oof], axis=0)
    e_auc = roc_auc_score(y, ens); e_ap = average_precision_score(y, ens)
    e_thr, e_f1 = best_f1_threshold(y, ens)
    results["ensemble"] = {"roc_auc": e_auc, "pr_auc": e_ap, "f1": e_f1, "thr": float(e_thr)}
    print(f"{'ENSEMBLE':>8} {e_auc:8.4f} {e_ap:8.4f} {e_f1:6.3f}")
    md.append(f"| **ensemble** | **{e_auc:.4f}** | {e_ap:.4f} | {e_f1:.3f} |")

    best = max(results, key=lambda k: results[k]["roc_auc"])
    md.append(f"\n**Best by ROC-AUC: `{best}` = {results[best]['roc_auc']:.4f}.**\n")

    cm = confusion_matrix(y, (ens >= e_thr).astype(int))
    md.append(f"\n## OOF confusion matrix (ensemble @ F1-optimal thr={e_thr:.3f})\n```\n{cm}\n```\n")
    md.append("```\n" + classification_report(y, (ens >= e_thr).astype(int),
              target_names=["Stay", "Churn"], zero_division=0) + "\n```\n")

    imp = pd.Series(importances, index=feats).sort_values(ascending=False)
    md.append("\n## Top 20 features (LightGBM)\n```\n" + imp.head(20).round(1).to_string() + "\n```\n")
    print("\nTop 12 features:\n", imp.head(12).round(1).to_string())

    # ---- figures for the paper ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, precision_recall_curve as prc
    sns.set_theme(style="whitegrid")
    FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    for k in list(oof) + ["ensemble"]:
        pk = ens if k == "ensemble" else oof[k]
        fpr, tpr, _ = roc_curve(y, pk)
        plt.plot(fpr, tpr, lw=1.5, label=f"{k} ({roc_auc_score(y, pk):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=.8); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC curves (OOF)"); plt.legend(fontsize=8)
    plt.subplot(1, 2, 2)
    for k in list(oof) + ["ensemble"]:
        pk = ens if k == "ensemble" else oof[k]
        pr, rc, _ = prc(y, pk)
        plt.plot(rc, pr, lw=1.5, label=f"{k} ({average_precision_score(y, pk):.3f})")
    plt.axhline(y.mean(), color="k", ls="--", lw=.8, label=f"base {y.mean():.2f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall (OOF)"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "roc_pr_curves.png", dpi=110, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(7, 5))
    sns.barplot(x=imp.head(15).values, y=imp.head(15).index, palette="mako")
    plt.title("Top 15 features (LightGBM importance)"); plt.xlabel("importance"); plt.ylabel("")
    plt.tight_layout(); plt.savefig(FIG / "feature_importance.png", dpi=110, bbox_inches="tight"); plt.close()

    Path("reports").mkdir(exist_ok=True)
    Path("reports/phase4_model.md").write_text("\n".join(md))
    json.dump(results, open("reports/metrics.json", "w"), indent=2)
    print("\nWrote reports/phase4_model.md and reports/metrics.json")
    return imp


if __name__ == "__main__":
    run()
