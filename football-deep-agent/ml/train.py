"""Training pipeline for football match prediction models.

Trains Logistic Regression, Random Forest, and XGBoost baselines
using time-based train/test split (no random shuffling).
Evaluates all three plus an averaging ensemble.

Uses efficient incremental feature computation instead of per-match
recalculation of league positions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import FEATURE_COLUMNS, recent_form, goals_avg, team_strength
from ml.preprocessing import PROCESSED_DIR
from ml import logistic_model, random_forest, xgboost_model, ensemble
from ml.evaluation import classification_metrics, probabilistic_metrics, brier_score_per_class

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
TRAIN_TEST_SPLIT_RATIO = 0.8


def load_processed() -> pd.DataFrame:
    path = PROCESSED_DIR / "matches_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No processed data at {path}. Run preprocessing first.")
    return pd.read_parquet(path)


def _compute_league_positions_up_to(df: pd.DataFrame, before_date) -> dict[str, int]:
    """Compute league positions for all teams using matches strictly before before_date."""
    mask = df["date"] < before_date
    prior = df[mask]
    if prior.empty:
        return {}

    all_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    team_points: dict[str, int] = {t: 0 for t in all_teams}

    for _, row in prior.iterrows():
        ht, at = row["home_team"], row["away_team"]
        hg, ag = row["home_goals"], row["away_goals"]
        if hg > ag:
            team_points[ht] += 3
        elif hg < ag:
            team_points[at] += 3
        else:
            team_points[ht] += 1
            team_points[at] += 1

    sorted_teams = sorted(team_points.keys(), key=lambda t: (-team_points[t], t))
    return {t: i + 1 for i, t in enumerate(sorted_teams)}


def build_training_table_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature rows efficiently using pre-sorted data."""
    rows = []
    total = len(df)

    for i, (_, match) in enumerate(df.iterrows()):
        if (i + 1) % 1000 == 0 or i == 0:
            print(f"  Building features: {i+1}/{total}", flush=True)

        date = match["date"]
        home, away = match["home_team"], match["away_team"]

        # Use the existing feature functions (they filter by date internally)
        feats = {
            "home_team_strength": team_strength(df, home, date),
            "away_team_strength": team_strength(df, away, date),
            "home_recent_form": recent_form(df, home, date),
            "away_recent_form": recent_form(df, away, date),
            "home_goals_avg": goals_avg(df, home, date, conceded=False),
            "away_goals_avg": goals_avg(df, away, date, conceded=False),
            "home_conceded_avg": goals_avg(df, home, date, conceded=True),
            "away_conceded_avg": goals_avg(df, away, date, conceded=True),
            "home_advantage": 1.0,
        }

        # League position: compute only at match boundaries (every 10th match)
        # to speed up. Use cached position for intermediate matches.
        if i % 10 == 0 or not rows:
            league_pos = _compute_league_positions_up_to(df, date)
        feats["league_position_difference"] = league_pos.get(home, 10) - league_pos.get(away, 10)

        feats["result"] = match["result"]
        rows.append(feats)

    return pd.DataFrame(rows)


def time_based_split(df: pd.DataFrame, ratio: float = TRAIN_TEST_SPLIT_RATIO):
    """Split by date: first `ratio` fraction for train, rest for test."""
    split_idx = int(len(df) * ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    print(f"Time-based split: train={len(train)}, test={len(test)}")
    return train, test


def train_all_models(train_df: pd.DataFrame) -> dict:
    """Train all three models and return them."""
    print("\n--- Training Logistic Regression ---")
    lr = logistic_model.train(train_df)
    logistic_model.save(lr)
    print("  Saved logistic_model.joblib")

    print("\n--- Training Random Forest ---")
    rf = random_forest.train(train_df)
    random_forest.save(rf)
    print("  Saved random_forest.joblib")

    print("\n--- Training XGBoost ---")
    xgb = xgboost_model.train(train_df)
    xgboost_model.save(xgb)
    print("  Saved xgboost_model.joblib")

    return {"logistic": lr, "random_forest": rf, "xgboost": xgb}


def _get_proba_aligned(model, X_test, name: str) -> tuple[np.ndarray, list[str]]:
    """Get probability array aligned to canonical class order [A, D, H]."""
    proba = model.predict_proba(X_test)
    canonical = ["A", "D", "H"]

    if name == "xgboost":
        # XGBoost uses label encoding H=0, D=1, A=2
        xgb_order = ["H", "D", "A"]
        col_map = {xgb_order[i]: i for i in range(3)}
        aligned = np.column_stack([proba[:, col_map[c]] for c in canonical])
        return aligned, canonical
    else:
        # sklearn models: model.classes_ gives actual class labels
        model_classes = list(model.classes_)
        col_map = {model_classes[i]: i for i in range(len(model_classes))}
        aligned = np.column_stack([proba[:, col_map[c]] for c in canonical])
        return aligned, canonical


def evaluate_models(models: dict, test_df: pd.DataFrame):
    """Evaluate each model and the ensemble on the test set."""
    X_test = test_df[FEATURE_COLUMNS]
    test_labels = test_df["result"]
    canonical = ["A", "D", "H"]

    results = {}

    for name, model in models.items():
        proba, classes = _get_proba_aligned(model, X_test, name)
        y_pred = [classes[i] for i in np.argmax(proba, axis=1)]
        y_true_encoded = [list(classes).index(c) for c in test_labels]

        clf_metrics = classification_metrics(test_labels, pd.Series(y_pred))
        prob_metrics = probabilistic_metrics(y_true_encoded, proba, list(classes))
        brier = {}
        for cls in classes:
            brier[f"brier_{cls}"] = brier_score_per_class(
                test_labels, pd.Series(proba[:, list(classes).index(cls)]), cls
            )
        results[name] = {**clf_metrics, **prob_metrics, **brier}

    # Ensemble
    print("\n--- Evaluating Ensemble ---")
    all_aligned = []
    for name, model in models.items():
        aligned, _ = _get_proba_aligned(model, X_test, name)
        all_aligned.append(aligned)

    ensemble_proba = np.mean(all_aligned, axis=0)
    y_pred_ensemble = [canonical[i] for i in np.argmax(ensemble_proba, axis=1)]
    y_true_encoded = [list(canonical).index(c) for c in test_labels]

    clf_metrics = classification_metrics(test_labels, pd.Series(y_pred_ensemble))
    prob_metrics = probabilistic_metrics(y_true_encoded, ensemble_proba, list(canonical))
    brier = {}
    for cls in canonical:
        brier[f"brier_{cls}"] = brier_score_per_class(
            test_labels, pd.Series(ensemble_proba[:, list(canonical).index(cls)]), cls
        )
    results["ensemble"] = {**clf_metrics, **prob_metrics, **brier}
    return results


def print_results(results: dict):
    """Print a comparison table."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "log_loss",
               "brier_H", "brier_D", "brier_A"]
    header = f"{'Metric':<20} {'Logistic':>10} {'RF':>10} {'XGBoost':>10} {'Ensemble':>10}"
    print(header)
    print("-" * 80)

    for m in metrics:
        vals = []
        for name in ["logistic", "random_forest", "xgboost", "ensemble"]:
            v = results.get(name, {}).get(m, float("nan"))
            vals.append(f"{v:>10.4f}")
        print(f"{m:<20} {' '.join(vals)}")

    print("=" * 80)

    lr_ll = results["logistic"]["log_loss"]
    rf_ll = results["random_forest"]["log_loss"]
    xgb_ll = results["xgboost"]["log_loss"]
    ens_ll = results["ensemble"]["log_loss"]
    best_single = min(lr_ll, rf_ll, xgb_ll)

    print(f"\nBest single model log loss: {best_single:.4f}")
    print(f"Ensemble log loss:          {ens_ll:.4f}")
    if ens_ll <= best_single:
        print("Ensemble matches or beats best single model!")
    else:
        print(f"Ensemble is {ens_ll - best_single:.4f} worse than best single model")


def main():
    print("Loading processed data...")
    df = load_processed()
    print(f"Loaded {len(df)} matches")

    print("\nBuilding training table...")
    full_table = build_training_table_fast(df)
    full_table["date"] = df["date"].values

    print("\nSplitting data (time-based)...")
    train_df, test_df = time_based_split(full_table)

    train_features = train_df[FEATURE_COLUMNS + ["result"]]
    test_features = test_df

    print("\nTraining models...")
    models = train_all_models(train_features)

    print("\nEvaluating models on test set...")
    results = evaluate_models(models, test_features)

    print_results(results)

    results_path = ARTIFACTS_DIR / "training_results.txt"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        for name, metrics in results.items():
            f.write(f"\n--- {name} ---\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
