# ML Prediction Engine Documentation

## Dataset Source

**Source**: Football-Data.co.uk Premier League Dataset
**Date Range**: 2020-2024 seasons
**Competition**: English Premier League
**Matches**: ~1,520 matches (380 per season × 4 seasons)

### Data Format

The raw dataset contains the following columns:
- `date`: Match date (YYYY-MM-DD format)
- `home_team`: Home team name
- `away_team`: Away team name
- `home_goals`: Goals scored by home team
- `away_goals`: Goals scored by away team

### Data Cleaning Steps

1. **Date Parsing**: Convert string dates to datetime objects
2. **Missing Value Handling**: Remove matches with missing goal data
3. **Result Derivation**: Calculate match result (H/D/A) from goal columns
4. **Sorting**: Order matches chronologically for time-based validation

## Feature Definitions

All features follow the **anti-leakage rule**: only information available BEFORE kickoff is used.

### 1. Team Strength (`home_team_strength`, `away_team_strength`)
- **Definition**: Win rate over last 20 matches
- **Calculation**: Wins / Total matches in window
- **Range**: 0.0 (no wins) to 1.0 (all wins)
- **Fallback**: 0.5 if no history available

### 2. Recent Form (`home_recent_form`, `away_recent_form`)
- **Definition**: Points per game over last 5 matches
- **Calculation**: (3 × wins + 1 × draws) / 5
- **Range**: 0.0 (all losses) to 3.0 (all wins)
- **Rationale**: Captures current team momentum

### 3. Goals Scored Average (`home_goals_avg`, `away_goals_avg`)
- **Definition**: Average goals scored per match over last 10 games
- **Calculation**: Total goals scored / 10
- **Range**: 0.0 to typically 3-4 goals
- **Usage**: Offensive strength indicator

### 4. Goals Conceded Average (`home_conceded_avg`, `away_conceded_avg`)
- **Definition**: Average goals conceded per match over last 10 games
- **Calculation**: Total goals conceded / 10
- **Range**: 0.0 to typically 2-3 goals
- **Usage**: Defensive strength indicator

### 5. Home Advantage (`home_advantage`)
- **Definition**: Binary indicator for home team
- **Value**: 1.0 for home team, 0.0 for away team
- **Rationale**: Home teams win ~46% of matches in major leagues

### 6. League Position Difference (`league_position_difference`)
- **Definition**: Difference in league standings (home_pos - away_pos)
- **Calculation**: Home team league position - Away team league position
- **Range**: -19 to +19 (in 20-team league)
- **Usage**: Relative team strength indicator

## Anti-Leakage Implementation

### Verification Test

The test in `tests/test_features.py` proves:
- Features for match on 2024-02-15 only use matches before that date
- A match on 2024-03-01 does NOT influence features for 2024-02-15
- No future information leaks into training data

### Implementation Details

```python
def _team_matches_before(df, team, before_date, n=None):
    """All matches for team STRICTLY before before_date."""
    mask = ((df["home_team"] == team) | (df["away_team"] == team)) & (df["date"] < before_date)
    prior = df.loc[mask].sort_values("date")
    return prior.tail(n) if n else prior
```

**Key**: The `< before_date` condition ensures no future matches are included.

## Model Architecture

### Baseline Models

1. **Logistic Regression** (`ml/logistic_model.py`)
   - Multinomial logistic regression for H/D/A classification
   - Max iterations: 1000
   - Good baseline, fast training

2. **Random Forest** (`ml/random_forest.py`)
   - 300 estimators, max depth 6
   - Handles non-linear relationships
   - Reduces overfitting with depth limit

3. **XGBoost** (`ml/xgboost_model.py`)
   - 300 estimators, max depth 4
   - Learning rate: 0.05
   - State-of-the-art gradient boosting

### Ensemble Method

**Simple Averaging**: Average class probabilities across all three models
- Equal weight to each model
- Reduces variance and overfitting
- Easy to interpret and debug

## Training and Evaluation

### Time-Based Split

- **Training Set**: Matches before 2024-01-01
- **Test Set**: Matches from 2024-01-01 onwards
- **No Random Shuffling**: Preserves temporal ordering

### Evaluation Metrics

1. **Classification Metrics**:
   - Accuracy: Overall correct predictions
   - Precision (macro): Average precision across H/D/A
   - Recall (macro): Average recall across H/D/A
   - F1 Score (macro): Harmonic mean of precision/recall

2. **Probabilistic Metrics**:
   - Log Loss: Measures prediction confidence
   - Brier Score: Measures probability calibration

## Known Issues and Fixes

### Issue 1: League Position Feature
**Problem**: `league_position_difference` defaults to 0.0
**Fix Needed**: Implement real league table lookup before match date
**Status**: Placeholder implementation

### Issue 2: No Sample Data
**Problem**: No raw data in `data/raw/` directory
**Fix**: User must download dataset from Football-Data.co.uk
**Alternative**: Create synthetic sample data for testing

### Issue 3: Model Artifacts
**Problem**: No trained models in `ml/artifacts/`
**Fix**: Run training script after data is available
**Command**: `python -m ml.train` (to be implemented)

## Usage Examples

### Single Match Prediction

```python
from ml.predict import predict_match

result = predict_match("Arsenal", "Chelsea")
print(result)
# {"home_win": 0.58, "draw": 0.24, "away_win": 0.18}
```

### Batch Evaluation

```python
from ml.evaluation import classification_metrics, probabilistic_metrics

y_true = [...]  # True results
y_pred = [...]  # Predicted results
y_proba = [...] # Predicted probabilities

metrics = classification_metrics(y_true, y_pred)
prob_metrics = probabilistic_metrics(y_true, y_proba, ["H", "D", "A"])
```

## Future Improvements

1. **Elo Ratings**: Replace win-rate with Elo-based strength ratings
2. **xG Integration**: Use expected goals for better feature quality
3. **Player Data**: Add player availability and form features
4. **Live Data**: Connect to real-time match data API
5. **Advanced Ensemble**: Stack models instead of simple averaging
