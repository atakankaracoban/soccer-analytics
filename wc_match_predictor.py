#!/usr/bin/env python3
"""Unified World Cup single-match predictor.

Core model:
  - 1500-base Elo ratings, replayed chronologically from international results.
  - Goal-difference multiplier and granular K factors.
  - Learned Poisson xG model using pre-match Elo snapshots.
  - Fast rolling form, opponent-adjusted form, and H2H features.
  - Optional calibrated Random Forest layer when scikit-learn is installed.

Example:
  python wc_match_predictor.py "South Africa" "Canada" --report
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import os
import pickle
import random
import ssl
import sys
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date
from pathlib import Path


DATA_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/results.csv"
)

ELO_BASE = 1500.0
HOME_ADV_ELO = 75.0
ELO_PROB_SCALE = 400.0
POISSON_ELO_SCALE = 400.0
FORM_WINDOW = 10
H2H_WINDOW = 10
DEFAULT_TRIALS = 50000

FEATURE_COLS = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "home_form",
    "away_form",
    "form_diff",
    "home_adj_form",
    "away_adj_form",
    "adj_form_diff",
    "h2h_home",
    "neutral",
    "tournament_w",
    "elo_win_prob",
    "poisson_home",
    "poisson_draw",
    "poisson_away",
]


TEAM_ALIASES = {
    "rsa": "South Africa",
    "south africa": "South Africa",
    "can": "Canada",
    "canada": "Canada",
    "usa": "United States",
    "united states": "United States",
    "cote d'ivoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "curacao": "Curacao",
    "curaçao": "Curacao",
    "bosnia": "Bosnia and Herzegovina",
    "bosnia & herz.": "Bosnia and Herzegovina",
    "dr congo": "DR Congo",
}


def k_factor(tournament: str) -> float:
    t = tournament.lower()
    if "fifa world cup" in t and "qualif" not in t:
        return 60.0
    if any(
        key in t
        for key in (
            "copa america",
            "copa america",
            "uefa euro",
            "africa cup",
            "afc asian cup",
            "gold cup",
            "concacaf nations",
        )
    ):
        return 50.0
    if "qualif" in t or "qualification" in t:
        return 40.0
    if "nations league" in t or "confederation" in t:
        return 35.0
    return 20.0


def tournament_weight(tournament: str) -> float:
    return k_factor(tournament) / 60.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / ELO_PROB_SCALE))


def canonical_team(name: str) -> str:
    stripped = name.strip()
    return TEAM_ALIASES.get(stripped.lower(), stripped)


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def load_results(cache_dir: Path, refresh: bool = False) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "international_results.csv"

    if cache_path.exists() and not refresh:
        raw = cache_path.read_text(encoding="utf-8")
    else:
        print("Downloading international results dataset...")
        try:
            with urllib.request.urlopen(DATA_URL, timeout=60, context=ssl_context()) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as exc:
            if cache_path.exists():
                print(f"Download failed; using cached results: {exc}")
                raw = cache_path.read_text(encoding="utf-8")
            else:
                raise
        cache_path.write_text(raw, encoding="utf-8")

    rows: list[dict] = []
    reader = csv.DictReader(io.StringIO(raw))
    for row in reader:
        try:
            home_score = int(float(row["home_score"]))
            away_score = int(float(row["away_score"]))
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_score": home_score,
                "away_score": away_score,
                "tournament": row.get("tournament", ""),
                "neutral": str(row.get("neutral", "")).upper() == "TRUE",
            }
        )

    rows.sort(key=lambda r: r["date"])
    return rows


def mean_or_half(values: deque[float]) -> float:
    return sum(values) / len(values) if values else 0.5


def weighted_mean_or_half(values: deque[tuple[float, float]]) -> float:
    if not values:
        return 0.5
    total_w = sum(weight for _, weight in values)
    if total_w <= 0:
        return 0.5
    return sum(score * weight for score, weight in values) / total_w


def h2h_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def h2h_rate(pair_log: deque[str], team: str) -> float:
    if not pair_log:
        return 0.5
    wins = sum(1 for winner in pair_log if winner == team)
    return wins / len(pair_log)


def goal_diff_multiplier(goal_diff: int) -> float:
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return 1.75


@dataclass
class RatingsBundle:
    ratings: dict[str, float]
    feature_rows: list[dict]
    poisson_rows: list[tuple[float, float, float]]
    match_count: int
    latest_date: str
    team_logs: dict[str, deque[dict]]
    h2h_logs: dict[tuple[str, str], deque[dict]]


def replay_matches(rows: list[dict]) -> RatingsBundle:
    ratings: dict[str, float] = {}
    feature_rows: list[dict] = []
    poisson_rows: list[tuple[float, float, float]] = []

    form: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    adj_form: dict[str, deque[tuple[float, float]]] = defaultdict(
        lambda: deque(maxlen=FORM_WINDOW)
    )
    h2h_recent: dict[tuple[str, str], deque[str]] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))
    team_logs: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=30))
    h2h_logs: dict[tuple[str, str], deque[dict]] = defaultdict(lambda: deque(maxlen=50))

    def rating(team: str) -> float:
        return ratings.setdefault(team, ELO_BASE)

    for row in rows:
        home = row["home_team"]
        away = row["away_team"]
        hs = row["home_score"]
        away_score = row["away_score"]
        neutral = row["neutral"]
        home_elo = rating(home)
        away_elo = rating(away)

        home_bonus = 0.0 if neutral else HOME_ADV_ELO
        elo_win_prob = expected_score(home_elo + home_bonus, away_elo)

        ph, pd, pa, _, _ = poisson_probabilities_from_xg(
            fallback_expected_goals(home_elo, away_elo, neutral)
        )

        home_form = mean_or_half(form[home])
        away_form = mean_or_half(form[away])
        home_adj_form = weighted_mean_or_half(adj_form[home])
        away_adj_form = weighted_mean_or_half(adj_form[away])
        pair_key = h2h_key(home, away)
        h2h_home = h2h_rate(h2h_recent[pair_key], home)

        result = 2 if hs > away_score else 1 if hs == away_score else 0
        feature_rows.append(
            {
                "elo_diff": home_elo - away_elo,
                "home_elo": home_elo,
                "away_elo": away_elo,
                "home_form": home_form,
                "away_form": away_form,
                "form_diff": home_form - away_form,
                "home_adj_form": home_adj_form,
                "away_adj_form": away_adj_form,
                "adj_form_diff": home_adj_form - away_adj_form,
                "h2h_home": h2h_home,
                "neutral": 1.0 if neutral else 0.0,
                "tournament_w": tournament_weight(row["tournament"]),
                "elo_win_prob": elo_win_prob,
                "poisson_home": ph,
                "poisson_draw": pd,
                "poisson_away": pa,
                "result": result,
            }
        )

        host_flag = 0.0 if neutral else 1.0
        poisson_rows.append(((home_elo - away_elo) / POISSON_ELO_SCALE, host_flag, hs))
        poisson_rows.append(((away_elo - home_elo) / POISSON_ELO_SCALE, 0.0, away_score))

        expected_home = elo_win_prob
        actual_home = 1.0 if hs > away_score else 0.5 if hs == away_score else 0.0
        actual_away = 1.0 - actual_home
        k = k_factor(row["tournament"]) * goal_diff_multiplier(abs(hs - away_score))
        ratings[home] = home_elo + k * (actual_home - expected_home)
        ratings[away] = away_elo + k * (actual_away - (1.0 - expected_home))

        home_pts = actual_home
        away_pts = actual_away
        form[home].append(home_pts)
        form[away].append(away_pts)
        adj_form[home].append((home_pts, away_elo / ELO_BASE))
        adj_form[away].append((away_pts, home_elo / ELO_BASE))

        winner = home if hs > away_score else away if away_score > hs else "Draw"
        h2h_recent[pair_key].append(winner)
        match_summary = {
            "date": row["date"],
            "score": f"{hs}-{away_score}",
            "home": home,
            "away": away,
            "winner": winner,
            "tournament": row["tournament"],
        }
        h2h_logs[pair_key].append(match_summary)

        for team, opp, gf, ga, pts in (
            (home, away, hs, away_score, home_pts),
            (away, home, away_score, hs, away_pts),
        ):
            team_logs[team].append(
                {
                    "date": row["date"],
                    "opponent": opp,
                    "gf": gf,
                    "ga": ga,
                    "result": "W" if pts == 1.0 else "D" if pts == 0.5 else "L",
                    "tournament": row["tournament"],
                    "neutral": neutral,
                }
            )

    return RatingsBundle(
        ratings=ratings,
        feature_rows=feature_rows,
        poisson_rows=poisson_rows,
        match_count=len(rows),
        latest_date=rows[-1]["date"] if rows else "",
        team_logs=team_logs,
        h2h_logs=h2h_logs,
    )


@dataclass
class PoissonGoalModel:
    intercept: float
    elo_coef: float
    home_coef: float
    iterations: int

    def expected_goals(
        self, elo_a: float, elo_b: float, neutral: bool = True
    ) -> tuple[float, float]:
        home_a = 0.0 if neutral else 1.0
        log_a = self.intercept + self.elo_coef * ((elo_a - elo_b) / POISSON_ELO_SCALE) + self.home_coef * home_a
        log_b = self.intercept + self.elo_coef * ((elo_b - elo_a) / POISSON_ELO_SCALE)
        return math.exp(max(-4.0, min(3.0, log_a))), math.exp(max(-4.0, min(3.0, log_b)))


def solve_3x3(a: list[list[float]], b: list[float]) -> list[float]:
    mat = [a[i][:] + [b[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot][col]) < 1e-12:
            raise ValueError("Singular matrix while fitting Poisson model")
        mat[col], mat[pivot] = mat[pivot], mat[col]
        div = mat[col][col]
        for j in range(col, 4):
            mat[col][j] /= div
        for r in range(3):
            if r == col:
                continue
            factor = mat[r][col]
            for j in range(col, 4):
                mat[r][j] -= factor * mat[col][j]
    return [mat[i][3] for i in range(3)]


def fit_poisson_model(rows: list[tuple[float, float, float]], max_iter: int = 35) -> PoissonGoalModel:
    avg_goals = sum(y for _, _, y in rows) / max(len(rows), 1)
    beta = [math.log(max(avg_goals, 0.05)), 0.12, 0.12]
    ridge = 1e-6

    for iteration in range(1, max_iter + 1):
        xtwx = [[0.0, 0.0, 0.0] for _ in range(3)]
        xtwz = [0.0, 0.0, 0.0]

        for diff, home, y in rows:
            x = (1.0, diff, home)
            eta = max(-5.0, min(3.0, beta[0] + beta[1] * diff + beta[2] * home))
            mu = max(math.exp(eta), 1e-8)
            z = eta + (y - mu) / mu
            for i in range(3):
                xtwz[i] += x[i] * mu * z
                for j in range(3):
                    xtwx[i][j] += x[i] * mu * x[j]

        for i in range(3):
            xtwx[i][i] += ridge
        new_beta = solve_3x3(xtwx, xtwz)
        delta = max(abs(new_beta[i] - beta[i]) for i in range(3))
        beta = new_beta
        if delta < 1e-7:
            return PoissonGoalModel(beta[0], beta[1], beta[2], iteration)

    return PoissonGoalModel(beta[0], beta[1], beta[2], max_iter)


def fallback_expected_goals(
    elo_a: float, elo_b: float, neutral: bool = True
) -> tuple[float, float]:
    home_bonus = 0.0 if neutral else HOME_ADV_ELO
    diff = (elo_a + home_bonus - elo_b) / ELO_PROB_SCALE
    mult = 10.0 ** diff
    ratio = min(max(math.sqrt(mult), 0.33), 3.0)
    total = 2.50
    xg_a = (total * ratio) / (1.0 + ratio)
    return xg_a, total - xg_a


def poisson_pmf(lam: float, max_goals: int = 10) -> list[float]:
    probs = [math.exp(-lam)]
    for k in range(1, max_goals + 1):
        probs.append(probs[-1] * lam / k)
    total = sum(probs)
    if total > 0:
        probs[-1] += 1.0 - total
    return probs


def poisson_probabilities_from_xg(
    xg: tuple[float, float], max_goals: int = 10
) -> tuple[float, float, float, str, dict[str, float]]:
    xg_a, xg_b = xg
    pa = poisson_pmf(xg_a, max_goals)
    pb = poisson_pmf(xg_b, max_goals)
    win_a = draw = win_b = 0.0
    scores: dict[str, float] = {}

    for ga, pga in enumerate(pa):
        for gb, pgb in enumerate(pb):
            p = pga * pgb
            key = f"{ga}-{gb}"
            scores[key] = p
            if ga > gb:
                win_a += p
            elif ga == gb:
                draw += p
            else:
                win_b += p

    outcome = "a" if win_a >= draw and win_a >= win_b else "draw" if draw >= win_b else "b"
    filtered = {
        score: prob
        for score, prob in scores.items()
        if (outcome == "a" and int(score.split("-")[0]) > int(score.split("-")[1]))
        or (outcome == "draw" and int(score.split("-")[0]) == int(score.split("-")[1]))
        or (outcome == "b" and int(score.split("-")[0]) < int(score.split("-")[1]))
    }
    most_likely = max(filtered.items(), key=lambda kv: kv[1])[0]
    return win_a, draw, win_b, most_likely, scores


def latest_form(bundle: RatingsBundle, team: str, n: int = FORM_WINDOW) -> tuple[float, float, dict]:
    log = list(bundle.team_logs.get(team, []))[-n:]
    if not log:
        return 0.5, 0.5, {"w": 0, "d": 0, "l": 0, "avg_gf": 0.0, "avg_ga": 0.0, "gd": 0.0}

    pts = []
    weighted = []
    for m in log:
        score = 1.0 if m["result"] == "W" else 0.5 if m["result"] == "D" else 0.0
        opp_rating = bundle.ratings.get(m["opponent"], ELO_BASE)
        pts.append(score)
        weighted.append((score, opp_rating / ELO_BASE))

    total_w = sum(w for _, w in weighted)
    adj = sum(s * w for s, w in weighted) / total_w if total_w else 0.5
    gf = sum(m["gf"] for m in log)
    ga = sum(m["ga"] for m in log)
    stats = {
        "w": sum(1 for m in log if m["result"] == "W"),
        "d": sum(1 for m in log if m["result"] == "D"),
        "l": sum(1 for m in log if m["result"] == "L"),
        "avg_gf": gf / len(log),
        "avg_ga": ga / len(log),
        "gd": (gf - ga) / len(log),
    }
    return sum(pts) / len(pts), adj, stats


def latest_h2h(bundle: RatingsBundle, team_a: str, team_b: str) -> tuple[float, list[dict]]:
    log = list(bundle.h2h_logs.get(h2h_key(team_a, team_b), []))
    if not log:
        return 0.5, []
    wins = sum(1 for m in log if m["winner"] == team_a)
    return wins / len(log), log


def vector_for_match(
    bundle: RatingsBundle,
    poisson: PoissonGoalModel,
    team_a: str,
    team_b: str,
    neutral: bool,
    tournament: str,
) -> tuple[list[float], dict]:
    elo_a = bundle.ratings.get(team_a, ELO_BASE)
    elo_b = bundle.ratings.get(team_b, ELO_BASE)
    form_a, adj_a, stats_a = latest_form(bundle, team_a)
    form_b, adj_b, stats_b = latest_form(bundle, team_b)
    h2h_a, h2h_log = latest_h2h(bundle, team_a, team_b)
    home_bonus = 0.0 if neutral else HOME_ADV_ELO
    elo_win_prob = expected_score(elo_a + home_bonus, elo_b)
    xg = poisson.expected_goals(elo_a, elo_b, neutral=neutral)
    p_home, p_draw, p_away, score, score_probs = poisson_probabilities_from_xg(xg)

    features = [
        elo_a - elo_b,
        elo_a,
        elo_b,
        form_a,
        form_b,
        form_a - form_b,
        adj_a,
        adj_b,
        adj_a - adj_b,
        h2h_a,
        1.0 if neutral else 0.0,
        tournament_weight(tournament),
        elo_win_prob,
        p_home,
        p_draw,
        p_away,
    ]
    context = {
        "elo_a": elo_a,
        "elo_b": elo_b,
        "form_a": form_a,
        "form_b": form_b,
        "adj_form_a": adj_a,
        "adj_form_b": adj_b,
        "stats_a": stats_a,
        "stats_b": stats_b,
        "h2h_a": h2h_a,
        "h2h_log": h2h_log,
        "xg_a": xg[0],
        "xg_b": xg[1],
        "poisson": (p_home, p_draw, p_away),
        "most_likely_score": score,
        "score_probs": score_probs,
    }
    return features, context


def softmax_blend(base: tuple[float, float, float], context: dict) -> tuple[float, float, float]:
    p_a, p_d, p_b = base
    form_edge = context["adj_form_a"] - context["adj_form_b"]
    h2h_edge = context["h2h_a"] - 0.5
    elo_edge = (context["elo_a"] - context["elo_b"]) / ELO_PROB_SCALE

    log_a = math.log(max(p_a, 1e-6)) + 0.45 * form_edge + 0.20 * h2h_edge + 0.12 * elo_edge
    log_d = math.log(max(p_d, 1e-6)) - 0.10 * abs(form_edge)
    log_b = math.log(max(p_b, 1e-6)) - 0.45 * form_edge - 0.20 * h2h_edge - 0.12 * elo_edge
    m = max(log_a, log_d, log_b)
    vals = [math.exp(log_a - m), math.exp(log_d - m), math.exp(log_b - m)]
    total = sum(vals)
    return vals[0] / total, vals[1] / total, vals[2] / total


def dataset_fingerprint(rows: list[dict]) -> str:
    if not rows:
        return "empty"
    basis = f"{len(rows)}:{rows[-1]['date']}:{rows[-1]['home_team']}:{rows[-1]['away_team']}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]


def train_or_load_rf(feature_rows: list[dict], cache_dir: Path, fingerprint: str):
    try:
        from sklearn.calibration import CalibratedClassifierCV  # type: ignore
        from sklearn.ensemble import RandomForestClassifier  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except Exception:
        return None, None, "scikit-learn not installed"

    cache_path = cache_dir / f"rf_calibrated_{fingerprint}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        return payload["model"], payload["scaler"], f"loaded cache {cache_path.name}"

    x = [[float(row[col]) for col in FEATURE_COLS] for row in feature_rows]
    y = [int(row["result"]) for row in feature_rows]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    base = RandomForestClassifier(
        n_estimators=260,
        max_depth=7,
        min_samples_leaf=6,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    try:
        model = CalibratedClassifierCV(estimator=base, cv=3, method="isotonic")
    except TypeError:
        model = CalibratedClassifierCV(base_estimator=base, cv=3, method="isotonic")
    model.fit(x_scaled, y)

    with cache_path.open("wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "features": FEATURE_COLS}, f)
    return model, scaler, f"trained and cached {cache_path.name}"


def rf_predict(model, scaler, features: list[float]) -> tuple[float, float, float] | None:
    if model is None or scaler is None:
        return None
    scaled = scaler.transform([features])
    probs = model.predict_proba(scaled)[0]
    classes = list(model.classes_)
    by_class = {int(cls): float(prob) for cls, prob in zip(classes, probs)}
    return by_class.get(2, 0.0), by_class.get(1, 0.0), by_class.get(0, 0.0)


def simulate_knockout_advancement(
    probs: tuple[float, float, float],
    elo_a: float,
    elo_b: float,
    trials: int,
    seed: int = 42,
) -> float:
    rng = random.Random(seed)
    p_a, p_d, _ = probs
    pen_edge = min(0.60, max(0.40, 0.50 + (elo_a - elo_b) / 2000.0))
    advances = 0
    for _ in range(trials):
        roll = rng.random()
        if roll < p_a:
            advances += 1
        elif roll < p_a + p_d and rng.random() < pen_edge:
            advances += 1
    return advances / trials


def mismatch_note(context: dict, probs: tuple[float, float, float], team_a: str, team_b: str) -> str | None:
    form_edge = context["adj_form_a"] - context["adj_form_b"]
    elo_edge = (context["elo_a"] - context["elo_b"]) / ELO_PROB_SCALE
    p_a, _, p_b = probs
    if form_edge > 0.20 and p_a < 0.40:
        return f"{team_a} has a clear adjusted-form edge, but the model still leans away because Elo/xG are stronger."
    if form_edge < -0.20 and p_b < 0.40:
        return f"{team_b} has a clear adjusted-form edge, but the model still leans away because Elo/xG are stronger."
    if abs(form_edge) > 0.25 and (form_edge * elo_edge) < 0:
        return "Form and Elo point in opposite directions; treat confidence as fragile."
    return None


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def make_report(
    team_a: str,
    team_b: str,
    context: dict,
    poisson_probs: tuple[float, float, float],
    final_probs: tuple[float, float, float],
    rf_probs: tuple[float, float, float] | None,
    rf_status: str,
    bundle: RatingsBundle,
    poisson_model: PoissonGoalModel,
    trials: int,
) -> str:
    p_a, p_d, p_b = final_probs
    adv_a = simulate_knockout_advancement(final_probs, context["elo_a"], context["elo_b"], trials)
    winner = team_a if p_a >= p_d and p_a >= p_b else "Draw" if p_d >= p_b else team_b
    note = mismatch_note(context, final_probs, team_a, team_b)
    h2h_log = context["h2h_log"][-6:]

    lines = [
        f"# Match prediction: {team_a} vs {team_b}",
        "",
        f"Generated: {date.today().isoformat()}",
        f"Data: {bundle.match_count:,} historical international matches through {bundle.latest_date}",
        f"RF layer: {rf_status}",
        "",
        "## Headline",
        "",
        f"- Prediction: {winner}",
        f"- Win/draw/win: {team_a} {pct(p_a)} | Draw {pct(p_d)} | {team_b} {pct(p_b)}",
        f"- Knockout advancement estimate: {team_a} {pct(adv_a)} | {team_b} {pct(1.0 - adv_a)}",
        f"- Expected goals: {team_a} {context['xg_a']:.2f} | {team_b} {context['xg_b']:.2f}",
        f"- Most likely score: {context['most_likely_score']} ({team_a}-{team_b})",
        "",
        "## Model signals",
        "",
        f"- Elo: {team_a} {context['elo_a']:.0f} | {team_b} {context['elo_b']:.0f} | diff {context['elo_a'] - context['elo_b']:+.0f}",
        f"- Raw form: {team_a} {context['form_a']:.3f} | {team_b} {context['form_b']:.3f}",
        f"- Opponent-adjusted form: {team_a} {context['adj_form_a']:.3f} | {team_b} {context['adj_form_b']:.3f}",
        f"- H2H win rate for {team_a}: {pct(context['h2h_a'])} across {len(context['h2h_log'])} recorded matches",
        "",
        "## Probability layers",
        "",
        f"- Poisson xG layer: {team_a} {pct(poisson_probs[0])} | Draw {pct(poisson_probs[1])} | {team_b} {pct(poisson_probs[2])}",
    ]
    if rf_probs is not None:
        lines.append(
            f"- Calibrated RF layer: {team_a} {pct(rf_probs[0])} | Draw {pct(rf_probs[1])} | {team_b} {pct(rf_probs[2])}"
        )
    lines.append(
        f"- Final blend: {team_a} {pct(final_probs[0])} | Draw {pct(final_probs[1])} | {team_b} {pct(final_probs[2])}"
    )

    lines.extend(
        [
            "",
            "## Recent form",
            "",
            f"- {team_a}: {context['stats_a']['w']}-{context['stats_a']['d']}-{context['stats_a']['l']} in last {FORM_WINDOW}; GF {context['stats_a']['avg_gf']:.2f}, GA {context['stats_a']['avg_ga']:.2f}, GD {context['stats_a']['gd']:+.2f}",
            f"- {team_b}: {context['stats_b']['w']}-{context['stats_b']['d']}-{context['stats_b']['l']} in last {FORM_WINDOW}; GF {context['stats_b']['avg_gf']:.2f}, GA {context['stats_b']['avg_ga']:.2f}, GD {context['stats_b']['gd']:+.2f}",
            "",
            "## H2H sample",
            "",
        ]
    )
    if h2h_log:
        for match in h2h_log:
            lines.append(
                f"- {match['date']}: {match['home']} {match['score']} {match['away']} ({match['tournament']})"
            )
    else:
        lines.append("- No recorded H2H matches in the dataset.")

    if note:
        lines.extend(["", "## Mismatch flag", "", f"- {note}"])

    lines.extend(
        [
            "",
            "## Method note",
            "",
            f"Elo starts at {int(ELO_BASE)}, not 1000, so public-facing ratings are comparable with standard Elo conventions. The Poisson model uses Elo differences scaled by 400, the same scale used by the Elo expected-score formula, which removes the hidden dependency between rating baseline and xG coefficients.",
            f"The learned Poisson equation is log(xG) = {poisson_model.intercept:.4f} + {poisson_model.elo_coef:.4f} * (EloDiff/400) + {poisson_model.home_coef:.4f} * home.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified World Cup single-match predictor")
    parser.add_argument("team_a", nargs="?", default="South Africa")
    parser.add_argument("team_b", nargs="?", default="Canada")
    parser.add_argument("--cache-dir", default=".wc_cache")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--neutral", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tournament", default="FIFA World Cup")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--report", action="store_true", help="Write a markdown report")
    parser.add_argument("--report-path", default="")
    args = parser.parse_args(argv)

    team_a = canonical_team(args.team_a)
    team_b = canonical_team(args.team_b)
    cache_dir = Path(args.cache_dir)

    rows = load_results(cache_dir, refresh=args.refresh)
    bundle = replay_matches(rows)
    poisson_model = fit_poisson_model(bundle.poisson_rows)
    fingerprint = dataset_fingerprint(rows)
    rf_model, scaler, rf_status = train_or_load_rf(bundle.feature_rows, cache_dir, fingerprint)

    features, context = vector_for_match(
        bundle, poisson_model, team_a, team_b, args.neutral, args.tournament
    )
    poisson_probs = context["poisson"]
    rf_probs = rf_predict(rf_model, scaler, features)
    if rf_probs is None:
        final_probs = softmax_blend(poisson_probs, context)
    else:
        adjusted = softmax_blend(poisson_probs, context)
        final_probs = tuple(0.62 * rf_probs[i] + 0.38 * adjusted[i] for i in range(3))  # type: ignore
        total = sum(final_probs)
        final_probs = tuple(p / total for p in final_probs)  # type: ignore

    report = make_report(
        team_a,
        team_b,
        context,
        poisson_probs,
        final_probs,  # type: ignore[arg-type]
        rf_probs,
        rf_status,
        bundle,
        poisson_model,
        max(1000, args.trials),
    )
    print(report)

    if args.report:
        if args.report_path:
            report_path = Path(args.report_path)
        else:
            safe_a = team_a.lower().replace(" ", "_")
            safe_b = team_b.lower().replace(" ", "_")
            report_path = Path("outputs") / f"{safe_a}_vs_{safe_b}_prediction.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(f"Saved report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
