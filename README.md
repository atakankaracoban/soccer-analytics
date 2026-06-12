# Soccer Analytics

Football analytics projects built with real event data and machine learning.
Part of a broader sports analytics portfolio focused on predictive modeling and system-level analysis.

---

## Projects

### World Cup 2026 Analytics
**Files:** `worldcup_analytics.py`, `statsbomb_bridge.py`

A two-layer predictive system for the 2026 FIFA World Cup.

**Data**
- 100 real matches from 2018 and 2022 FIFA World Cups via [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- Event-level data aggregated to match-level features per team
- 2026 squad statistics compiled from qualifier campaigns

**Model Architecture**
- Layer 1 — Match outcome prediction: Logistic Regression + Random Forest (binary: win/loss)
- Layer 2 — Player contribution scoring: per-90 normalized metrics with position-specific weights

**Feature Engineering**
All features are computed as differentials (team minus opponent) to capture relative strength:

| Feature | Description |
|---|---|
| `xG_diff` | Expected goals differential — strongest single predictor |
| `xGA_diff` | Defensive quality differential |
| `pos_diff` | Possession differential |
| `ppda_diff` | Pressing intensity differential (lower PPDA = more aggressive press) |
| `press_diff` | Raw pressing volume differential |
| `xg_quality_diff` | Shot quality differential (xG per shot) |

**Validation**
- 5-fold stratified cross-validation (walk-forward not applicable at match level)
- Logistic Regression CV accuracy: 0.650 ± 0.084
- Random Forest CV accuracy: 0.650 ± 0.114
- Brier Score: 0.197 (random baseline = 0.25)

**Baseline Comparison**

| Baseline | Accuracy |
|---|---|
| Coin flip | 0.500 |
| Majority class | 0.550 |
| Naive xG diff | 0.650 |
| Logistic Regression | 0.650 |
| Random Forest | 0.650 |

Current models match but do not exceed the naive xG differential baseline. This is an honest finding — xG captures most of the predictive signal in 100 matches. Expanding to multi-class (W/D/L) and adding historical World Cups (1986–2014) are the identified next steps.

**Calibration**
Calibration curve shows good alignment with diagonal across probability buckets, confirming predicted probabilities are reliable.

---

### Premier League Player Analysis
**File:** `football_analyst.py`

Position-adjusted player performance scoring for Premier League outfield players.

- Per-90 normalization across all metrics
- Non-penalty xG (npxG), xA, progressive carries, pressures
- Position-specific weight systems (FW / MF / DF)
- xG overperformance tracking

---

## Repository Structure

```
soccer_analytics/
├── worldcup_analytics.py       # Main WC 2026 model + dashboard
├── statsbomb_bridge.py         # StatsBomb event data pipeline
├── statsbomb_wc_train.csv      # 100-match training dataset (2018 + 2022 WC)
├── football_analyst.py         # Premier League player scoring
└── outputs/
    ├── wc_predictions_2026.csv
    ├── wc_team_stats_2026.csv
    ├── wc_player_scores_2026.csv
    └── worldcup_analytics_2026.png
```

---

## Setup

```bash
pip install statsbombpy scikit-learn pandas numpy matplotlib joblib
```

Run the World Cup model:
```bash
python worldcup_analytics.py
```

Rebuild training data from StatsBomb (takes ~3 min due to rate limiting):
```python
from statsbomb_bridge import build_real_training_data
df = build_real_training_data(use_cache=False)
```

---

## Known Limitations

- Binary classification only (W/L) — draws excluded from current model
- 100-match training set is small; model does not yet outperform xG naive baseline
- 2026 predictions use pre-tournament squad averages, not rolling in-tournament form
- No Vegas odds comparison yet (identified as next validation step)

---

## Next Steps

- [ ] Expand to W/D/L multi-class classification
- [ ] Add 1986–2014 World Cup data (StatsBomb open data available)
- [ ] Walk-forward validation by tournament year
- [ ] Compare against Vegas closing odds as external benchmark
- [ ] In-tournament rolling feature updates as 2026 group stage progresses
