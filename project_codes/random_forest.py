# random_forest.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from data_processing import ProcessedData
from pathlib import Path

# add path for visualizations
BASE_DIR = Path(__file__).resolve().parent.parent
PATH = BASE_DIR / "final_report" / "plots"

# ClassificationResult class — adapted from course notebook 
# Avoids repetition of code whenever we want to print final metric summary
class ClassificationResult:
    """Stores one model run: predictions, confusion matrix, and metrics."""

    LABELS = ["voted", "not_voted"]

    def __init__(self, name, y_true, y_pred):
        self.name = name
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.cm = confusion_matrix(y_true, y_pred)

    def metrics(self):
        tn, fp, fn, tp = self.cm.ravel()
        return {
            "precision": float(precision_score(self.y_true, self.y_pred, zero_division=0)),
            "recall":    float(recall_score(self.y_true, self.y_pred, zero_division=0)),
            "f1":        float(f1_score(self.y_true, self.y_pred, zero_division=0)),
            "fp": int(fp), "fn": int(fn),
            "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        }

    def summary(self):
        m = self.metrics()
        print(f"  {self.name:18s}  F1={m['f1']:.4f}  Recall={m['recall']:.4f}  "
              f"Prec={m['precision']:.4f}  FPR={m['fpr']:.4f}  FNR={m['fnr']:.4f}")
        return m

    def __repr__(self):
        m = self.metrics()
        return f"ClassificationResult({self.name!r}, F1={m['f1']:.4f})"


# Tuning functions to find number of trees
# We pick 500 as anything above 300 seems to have same effect
# Picked 500 instead of 300 to have some factor of safety
# Random Forest is also not affected by Overfitting
def tune_n_estimators(X_train, y_train, X_val, y_val):
    """Sweep n_estimators, plot validation AUC vs number of trees."""
    n_values = [50, 100, 200, 300, 500, 800, 1000]
    aucs = []
    for n in n_values:
        m = RandomForestClassifier(
            n_estimators=n, max_features="sqrt", class_weight="balanced",
            n_jobs=1, random_state=123)
        m.fit(X_train, y_train)
        auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        aucs.append(auc)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_values, aucs, marker="o", lw=2, color="steelblue")
    ax.set_xlabel("n_estimators", fontsize=12)
    ax.set_ylabel("Validation AUC", fontsize=12)
    ax.set_title("Performance vs. number of trees", fontsize=13)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.savefig(PATH / 'random_forest_performance.png')

# Use oob-error to find optimal trees
#same purpose as above but another metric
#this gave 100-300 are optimal, so 500 is a good choice here as well
def tune_n_estimators_oob(X_train, y_train):
    """Sweep n_estimators and plot OOB error vs number of trees."""
    n_values = [50, 100, 200, 300, 500, 800, 1000]
    oob_errors = []

    for n in n_values:
        m = RandomForestClassifier(
            n_estimators=n,
            max_features="sqrt",
            class_weight="balanced",
            oob_score=True,
            bootstrap=True,
            n_jobs=-1,
            random_state=123
        )
        m.fit(X_train, y_train)
        oob_errors.append(1 - m.oob_score_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_values, oob_errors, marker="o", lw=2, color="steelblue")
    ax.set_xlabel("n_estimators", fontsize=12)
    ax.set_ylabel("OOB error", fontsize=12)
    ax.set_title("OOB error vs. number of trees", fontsize=13)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.savefig(PATH / 'random_forest_obb.png')

    return n_values, oob_errors

#tuning function to identity number of random features at every node for prediction
#Based on F1 value 
def tune_max_features(X_train, y_train, X_val, y_val):
    """Sweep max_features, return best label and F1."""
    options = {"sqrt": "sqrt", "log2": "log2",
               "0.3": 0.3, "0.5": 0.5, "1.0": 1.0}
    best_label, best_f1 = None, 0
    for label, mf in options.items():
        m = RandomForestClassifier(
            n_estimators=500, max_features=mf, class_weight="balanced",
            n_jobs=1, random_state=123)
        m.fit(X_train, y_train)
        f1 = f1_score(y_val, m.predict(X_val))
        if f1 > best_f1:
            best_label, best_f1 = label, f1
    return best_label, best_f1

#find threshold to categorize output as non-voter
#threshold is where F1 is highest, followed by importance to Recall and Precision
def tune_threshold(rf, X_val, y_val, thresholds=None, verbose=True):
    """Sweep thresholds, print table if verbose, return best threshold and metrics."""
    if thresholds is None:
        thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    proba = rf.predict_proba(X_val)[:, 1]

    if verbose:
        print(f"  {'thresh':>7} {'recall':>7} {'prec':>7} {'F1':>7} "
              f"{'FPR':>7} {'FNR':>7} {'FP':>7} {'FN':>7}")
        print(f"  {'─'*7} {'─'*7} {'─'*7} {'─'*7} "
              f"{'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    best_t, best_f1, best_r, best_p = 0.5, 0, 0, 0
    sweep = {"thresh": [], "recall": [], "precision": [], "f1": []}

    for t in thresholds:
        pred = (proba >= t).astype(int)
        cm = confusion_matrix(y_val, pred)
        tn, fp, fn, tp = cm.ravel()
        r = recall_score(y_val, pred)
        p = precision_score(y_val, pred, zero_division=0)
        f = f1_score(y_val, pred)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        sweep["thresh"].append(t)
        sweep["recall"].append(r)
        sweep["precision"].append(p)
        sweep["f1"].append(f)

        if verbose:
            print(f"  {t:>7.2f} {r:>7.3f} {p:>7.3f} {f:>7.3f} "
                  f"{fpr:>7.3f} {fnr:>7.3f} {fp:>7d} {fn:>7d}")
        if f > best_f1:
            best_t, best_f1 = t, f
            best_r, best_p = r, p

    return best_t, best_f1, best_r, best_p, sweep

#plot threshold graph for visualization in report and presentation
def plot_threshold_sweep(sweep, best_t):
    """Plot F1, Precision, and Recall vs threshold for presentation file"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep["thresh"], sweep["f1"], marker="o", lw=2, label="F1", color="steelblue")
    ax.plot(sweep["thresh"], sweep["recall"], marker="s", lw=2, label="Recall", color="green")
    ax.plot(sweep["thresh"], sweep["precision"], marker="^", lw=2, label="Precision", color="orange")
    ax.axvline(best_t, ls="--", color="red", alpha=0.6, label=f"chosen t={best_t:.2f}")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("F1, Precision & Recall vs Threshold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.savefig(PATH / 'random_forest_threshold.png')

#find top features within features available using permutation
#the method uses validation dataset and shuffles feature column to see how it affects
#reduction in accuracy. More reduction means more importance of variable
def get_top_features(rf, X_val, y_val, feature_names, top_n=5):
    """Return top features by permutation importance."""
    perm = permutation_importance(rf, X_val, y_val,
            n_repeats=10, random_state=123, n_jobs=1)
    pim = perm.importances_mean
    order = pim.argsort()[::-1][:top_n]
    return [(feature_names[i], pim[i]) for i in order]


def main():

    ds = ProcessedData(random_state=123)
    bal = np.bincount(ds.train_y)
    print(f"Data: {bal[0]} voted, {bal[1]} non-voted ({ds.train_y.mean():.1%} non-voter)")

    # generate a plot to find no. of trees in the forest
    tune_n_estimators(ds.train_x, ds.train_y, ds.val_x, ds.val_y)
    tune_n_estimators_oob(ds.train_x, ds.train_y)
    # create random forest to predict output
    rf = RandomForestClassifier(
        n_estimators=500, max_features="sqrt", class_weight="balanced",
        n_jobs=1, random_state=123,
    ).fit(ds.train_x, ds.train_y)

    # generate baseline stats with threshold at 0.50
    base = ClassificationResult("Baseline (t=0.50)", ds.val_y, rf.predict(ds.val_x))
    base.summary()

    # output best F1 and corresponding label for random feature selection
    best_mf, best_mf_f1 = tune_max_features(
        ds.train_x, ds.train_y, ds.val_x, ds.val_y)
    print(f"  Best max_features: {best_mf} (F1={best_mf_f1:.4f})")

    # output table and graph to find best F1
    print(f"\n  Threshold table:")
    best_t, best_f1, best_r, best_p, sweep = tune_threshold(
        rf, ds.val_x, ds.val_y)
    print(f"  Best threshold: {best_t:.2f} (F1={best_f1:.4f}  "
          f"Recall={best_r:.4f}  Prec={best_p:.4f})")
    plot_threshold_sweep(sweep, best_t)

    #top features
    top = get_top_features(rf, ds.val_x, ds.val_y, ds.feature_order)
    print(f"  Top features: {top[0][0]} ({top[0][1]:.4f}), "
          f"{top[1][0]} ({top[1][1]:.4f})")

    #final test
    final_x = np.concatenate([ds.train_x, ds.val_x], axis=0)
    final_y = np.concatenate([ds.train_y, ds.val_y], axis=0)
    final_rf = RandomForestClassifier(
        n_estimators=500, max_features="sqrt", class_weight="balanced",
        n_jobs=1, random_state=123,
    ).fit(final_x, final_y)

    proba = final_rf.predict_proba(ds.test_x)[:, 1]
    pred = (proba >= best_t).astype(int)
    test_result = ClassificationResult("Final test", ds.test_y, pred)
    test_result.summary()


if __name__ == "__main__":
    main()