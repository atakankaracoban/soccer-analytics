# Match prediction: South Africa vs Canada

Generated: 2026-06-28
Data: 49,477 historical international matches through 2026-06-27
RF layer: loaded cache rf_calibrated_b4be7902969d.pkl

## Headline

- Prediction: Canada
- Win/draw/win: South Africa 22.8% | Draw 24.5% | Canada 52.8%
- Knockout advancement estimate: South Africa 33.3% | Canada 66.7%
- Expected goals: South Africa 0.92 | Canada 1.56
- Most likely score: 0-1 (South Africa-Canada)

## Model signals

- Elo: South Africa 1700 | Canada 1844 | diff -144
- Raw form: South Africa 0.450 | Canada 0.600
- Opponent-adjusted form: South Africa 0.432 | Canada 0.587
- H2H win rate for South Africa: 100.0% across 1 recorded matches

## Probability layers

- Poisson xG layer: South Africa 22.1% | Draw 25.5% | Canada 52.4%
- Calibrated RF layer: South Africa 23.3% | Draw 24.1% | Canada 52.6%
- Final blend: South Africa 22.8% | Draw 24.5% | Canada 52.8%

## Recent form

- South Africa: 3-3-4 in last 10; GF 0.90, GA 1.10, GD -0.20
- Canada: 3-6-1 in last 10; GF 1.50, GA 0.60, GD +0.90

## H2H sample

- 2007-11-20: South Africa 2-0 Canada (Friendly)

## Method note

Elo starts at 1500, not 1000, so public-facing ratings are comparable with standard Elo conventions. The Poisson model uses Elo differences scaled by 400, the same scale used by the Elo expected-score formula, which removes the hidden dependency between rating baseline and xG coefficients.
The learned Poisson equation is log(xG) = 0.1790 + 0.7381 * (EloDiff/400) + 0.3003 * home.
