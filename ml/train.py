from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.features import (  # noqa: E402
    METRICS_PATH,
    MODEL_PATH,
    fit_feature_spec,
    load_raw,
    make_target,
    save_feature_spec,
    transform,
)

SEED = 42

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "n_jobs": 1,
    "random_state": SEED,
}

MIN_CV_AUC = 0.70


def cross_validate_auc(model: xgb.XGBClassifier, features, target) -> tuple[float, float]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    for train_index, test_index in splitter.split(features, target):
        fold_model = clone(model)
        fold_model.fit(features.iloc[train_index], target[train_index])
        probabilities = fold_model.predict_proba(features.iloc[test_index])[:, 1]
        scores.append(roc_auc_score(target[test_index], probabilities))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> int:
    frame = load_raw()
    target = make_target(frame)
    spec = fit_feature_spec(frame)
    features = transform(frame, spec)

    model = xgb.XGBClassifier(**MODEL_PARAMS)
    cv_mean, cv_std = cross_validate_auc(model, features, target)

    train_features, holdout_features, train_target, holdout_target = train_test_split(
        features, target, test_size=0.2, random_state=SEED, stratify=target
    )
    model.fit(train_features, train_target)
    holdout_auc = float(
        roc_auc_score(holdout_target, model.predict_proba(holdout_features)[:, 1])
    )

    model = xgb.XGBClassifier(**MODEL_PARAMS)
    model.fit(features, target)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(MODEL_PATH))
    save_feature_spec(spec)

    gain = model.get_booster().get_score(importance_type="gain")
    top_features = sorted(gain.items(), key=lambda item: -item[1])[:10]
    metrics = {
        "rows": int(len(frame)),
        "features": int(features.shape[1]),
        "bad_rate": float(target.mean()),
        "cv_auc_mean": round(cv_mean, 4),
        "cv_auc_std": round(cv_std, 4),
        "holdout_auc": round(holdout_auc, 4),
        "threshold": 0.5,
        "params": MODEL_PARAMS,
        "top_features_by_gain": [
            {"feature": name, "gain": round(float(value), 4)} for name, value in top_features
        ],
        "model_path": str(MODEL_PATH.relative_to(MODEL_PATH.parent.parent)),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Rows: {metrics['rows']}  Features: {metrics['features']}  Bad rate: {metrics['bad_rate']:.3f}")
    print(f"CV AUC: {cv_mean:.4f} +/- {cv_std:.4f}   Holdout AUC: {holdout_auc:.4f}")
    print(f"Model: {MODEL_PATH}")
    print(f"Metrics: {METRICS_PATH}")

    if cv_mean < MIN_CV_AUC:
        print(
            f"WARNING: CV AUC {cv_mean:.4f} is below the {MIN_CV_AUC:.2f} floor - "
            "edge cases may be scarce or unreliable."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
