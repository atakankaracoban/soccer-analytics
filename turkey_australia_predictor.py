"""
World Cup Match Predictor
=========================
ELO-tabanlı uluslararası futbol maç tahmin sistemi.

Veri kaynağı:
  github.com/martj42/international_results
  49.000+ uluslararası maç, 1872'den günümüze, düzenli güncelleme.

Model:
  - ELO rating (tüm tarihsel maçlardan hesaplanır)
  - Recent form (son 10 maç)
  - Head-to-head geçmişi
  - Neutral ground flag
  - Tournament importance weight
  - Hedef: W / D / L (3-sınıf)

Kullanım:
  python worldcup_predictor.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

DATA_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/results.csv"
)

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup":                1.00,
    "UEFA Euro":                     0.85,
    "Copa América":                  0.85,
    "Africa Cup of Nations":         0.85,
    "AFC Asian Cup":                 0.85,
    "CONCACAF Gold Cup":             0.75,
    "FIFA World Cup qualification":  0.75,
    "UEFA Euro qualification":       0.65,
    "Copa América qualification":    0.60,
    "Friendly":                      0.30,
}

K_BASE       = 32
HOME_ADV_ELO = 100
FORM_WINDOW  = 10
H2H_WINDOW   = 10
FEATURE_COLS = [
    "elo_diff", "home_elo", "away_elo",
    "home_form", "away_form", "form_diff",
    "home_adj_form", "away_adj_form", "adj_form_diff",
    "h2h_home", "neutral", "tournament_w", "elo_win_prob"
]


def load_data(url=DATA_URL):
    print("Veri yükleniyor...")
    df = pd.read_csv(url)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["neutral"] = df["neutral"].astype(str).str.upper() == "TRUE"
    print(f"  {len(df):,} maç yüklendi ({df['date'].min().year}–{df['date'].max().year})")
    return df


def get_tournament_weight(tournament):
    for key, w in TOURNAMENT_WEIGHTS.items():
        if key in str(tournament):
            return w
    return 0.5


def compute_elo(df):
    elo_ratings = {}

    def get_elo(team):
        return elo_ratings.get(team, 1500.0)

    def expected(a, b):
        return 1 / (1 + 10 ** ((b - a) / 400))

    pre_elos = []
    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hs, as_   = row["home_score"], row["away_score"]
        neutral   = row["neutral"]
        K         = K_BASE * get_tournament_weight(row["tournament"])

        elo_h = get_elo(home)
        elo_a = get_elo(away)
        bonus = 0.0 if neutral else HOME_ADV_ELO
        exp_h = expected(elo_h + bonus, elo_a)

        score_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        score_a = 1.0 - score_h

        pre_elos.append({"home_elo": elo_h, "away_elo": elo_a})
        elo_ratings[home] = elo_h + K * (score_h - exp_h)
        elo_ratings[away] = elo_a + K * (score_a - (1 - exp_h))

    elo_df = pd.DataFrame(pre_elos, index=df.index)
    df_out = pd.concat([df, elo_df], axis=1)

    print("\nTop 10 ELO:")
    for i, (t, e) in enumerate(
        sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)[:10], 1
    ):
        print(f"  {i:2}. {t:<25} {e:.0f}")

    return elo_ratings, df_out


def compute_recent_form(df, team, before_date, n=FORM_WINDOW):
    matches = df[
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < before_date)
    ].tail(n)
    if len(matches) == 0:
        return 0.5
    pts = []
    for _, r in matches.iterrows():
        if r["home_team"] == team:
            pts.append(1.0 if r["home_score"] > r["away_score"]
                       else 0.5 if r["home_score"] == r["away_score"] else 0.0)
        else:
            pts.append(1.0 if r["away_score"] > r["home_score"]
                       else 0.5 if r["away_score"] == r["home_score"] else 0.0)
    return np.mean(pts)


def compute_adj_form(df, team, before_date, n=FORM_WINDOW):
    """
    Opponent-adjusted form skoru.

    Ham form yerine rakip ELO'suyla ağırlıklandırılmış skor kullanır.
    Güçlü rakibe karşı galibiyet, zayıf rakibe karşı galibiyetten daha çok sayar.

    Formül:
        adj_form = Σ(sonuç × rakip_elo/1500) / Σ(rakip_elo/1500)

    Neden gerekli:
        Türkiye 0.85 form → çoğunlukla Bulgaristan, Kosova, Gürcistan'a karşı
        Avustralya 0.55 form → Kanada, Yeni Zelanda, Kolombiya'ya karşı
        Ham form bu farkı görmez, adjusted form görür.
    """
    matches = df[
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < before_date)
    ].tail(n)
    if len(matches) == 0:
        return 0.5

    weighted_scores = []
    weights = []
    for _, r in matches.iterrows():
        is_home = r["home_team"] == team
        gf = r["home_score"] if is_home else r["away_score"]
        ga = r["away_score"] if is_home else r["home_score"]
        result = 1.0 if gf > ga else (0.5 if gf == ga else 0.0)

        opp_elo = r["away_elo"] if is_home else r["home_elo"]
        weight = opp_elo / 1500.0

        weighted_scores.append(result * weight)
        weights.append(weight)

    return sum(weighted_scores) / sum(weights) if weights else 0.5


def compute_h2h(df, team1, team2, before_date, n=H2H_WINDOW):
    h2h = df[
        (((df["home_team"] == team1) & (df["away_team"] == team2)) |
         ((df["home_team"] == team2) & (df["away_team"] == team1))) &
        (df["date"] < before_date)
    ].tail(n)
    if len(h2h) == 0:
        return 0.5
    wins = sum(
        1 for _, r in h2h.iterrows()
        if (r["home_team"] == team1 and r["home_score"] > r["away_score"]) or
           (r["away_team"] == team1 and r["away_score"] > r["home_score"])
    )
    return wins / len(h2h)


def build_features(df):
    print("\nFeature engineering...")
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 5000 == 0:
            print(f"  {i:,} / {len(df):,}")
        home, away = row["home_team"], row["away_team"]
        date       = row["date"]
        neutral    = row["neutral"]
        t_w        = get_tournament_weight(row["tournament"])
        elo_h, elo_a = row["home_elo"], row["away_elo"]
        bonus      = 0.0 if neutral else HOME_ADV_ELO
        elo_win_prob = 1 / (1 + 10 ** ((elo_a - (elo_h + bonus)) / 400))

        hs, as_ = row["home_score"], row["away_score"]
        result = 2 if hs > as_ else (1 if hs == as_ else 0)

        hf   = compute_recent_form(df, home, date)
        af   = compute_recent_form(df, away, date)
        haf  = compute_adj_form(df, home, date)
        aaf  = compute_adj_form(df, away, date)

        rows.append({
            "elo_diff":      elo_h - elo_a,
            "home_elo":      elo_h,
            "away_elo":      elo_a,
            "home_form":     hf,
            "away_form":     af,
            "form_diff":     hf - af,
            "home_adj_form": haf,
            "away_adj_form": aaf,
            "adj_form_diff": haf - aaf,
            "h2h_home":      compute_h2h(df, home, away, date),
            "neutral":       int(neutral),
            "tournament_w":  t_w,
            "elo_win_prob":  elo_win_prob,
            "result":        result,
        })

    df_feat = pd.DataFrame(rows)
    dist = df_feat["result"].value_counts(normalize=True)
    print(f"  Home win={dist.get(2,0):.1%} | Draw={dist.get(1,0):.1%} | Away win={dist.get(0,0):.1%}")
    return df_feat


def train_model(df_feat):
    print("\n--- Model Eğitimi ---")
    X = df_feat[FEATURE_COLS].values
    y = df_feat["result"].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr = LogisticRegression(multi_class="multinomial", max_iter=1000, random_state=42)
    lr_cv = cross_val_score(lr, X_s, y, cv=cv, scoring="accuracy")
    print(f"Logistic Regression:  {lr_cv.mean():.3f} ± {lr_cv.std():.3f}")

    rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                 random_state=42, n_jobs=-1)
    rf_cv = cross_val_score(rf, X_s, y, cv=cv, scoring="accuracy")
    print(f"Random Forest:        {rf_cv.mean():.3f} ± {rf_cv.std():.3f}")

    naive_home = (y == 2).mean()
    naive_elo  = (np.where(df_feat["elo_diff"] > 0, 2, 0) == y).mean()
    print(f"Naive (home win):     {naive_home:.3f}")
    print(f"Naive (ELO fav):      {naive_elo:.3f}")

    rf.fit(X_s, y)
    lr.fit(X_s, y)

    print("\nFeature Importance (RF):")
    for feat, imp in sorted(zip(FEATURE_COLS, rf.feature_importances_),
                            key=lambda x: x[1], reverse=True):
        print(f"  {feat:<20} {imp:.3f} {'█' * int(imp*80)}")

    rf_cal = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=300, max_depth=6,
                               min_samples_leaf=5, random_state=42),
        cv=5, method="isotonic"
    )
    rf_cal.fit(X_s, y)
    return rf_cal, lr, scaler


def predict_match(team1, team2, elo_ratings, df_full, rf_model, scaler,
                  neutral=True, tournament="FIFA World Cup"):
    elo1 = elo_ratings.get(team1, 1500)
    elo2 = elo_ratings.get(team2, 1500)
    today = pd.Timestamp.today()
    bonus = 0.0 if neutral else HOME_ADV_ELO
    elo_win_prob = 1 / (1 + 10 ** ((elo2 - (elo1 + bonus)) / 400))

    form1 = compute_recent_form(df_full, team1, today)
    form2 = compute_recent_form(df_full, team2, today)
    h2h   = compute_h2h(df_full, team1, team2, today)
    t_w   = get_tournament_weight(tournament)

    adj_form1 = compute_adj_form(df_full, team1, today)
    adj_form2 = compute_adj_form(df_full, team2, today)

    features_s = scaler.transform([[
        elo1 - elo2, elo1, elo2,
        form1, form2, form1 - form2,
        adj_form1, adj_form2, adj_form1 - adj_form2,
        h2h, int(neutral), t_w, elo_win_prob
    ]])

    probs = rf_model.predict_proba(features_s)[0]
    prob_dict = dict(zip(rf_model.classes_, probs))
    p_win  = prob_dict.get(2, 0)
    p_draw = prob_dict.get(1, 0)
    p_loss = prob_dict.get(0, 0)

    if p_win >= p_draw and p_win >= p_loss:
        prediction, confidence = f"{team1} wins", p_win
    elif p_draw >= p_win and p_draw >= p_loss:
        prediction, confidence = "Draw", p_draw
    else:
        prediction, confidence = f"{team2} wins", p_loss

    return {
        "team1": team1, "team2": team2,
        "team1_elo": round(elo1), "team2_elo": round(elo2),
        "team1_form": round(form1, 3), "team2_form": round(form2, 3),
        "h2h_team1": round(h2h, 3),
        f"{team1}_win": round(p_win, 3),
        "draw": round(p_draw, 3),
        f"{team2}_win": round(p_loss, 3),
        "prediction": prediction,
        "confidence": round(confidence * 100, 1),
    }


def print_prediction(res):
    t1, t2 = res["team1"], res["team2"]
    print(f"\n{'='*52}")
    print(f"  {t1}  vs  {t2}")
    print(f"{'='*52}")
    print(f"  ELO      : {t1} {res['team1_elo']}  |  {t2} {res['team2_elo']}")
    print(f"  Form     : {t1} {res['team1_form']}  |  {t2} {res['team2_form']}")
    print(f"  H2H      : {t1} win rate {res['h2h_team1']:.0%}")
    print(f"  ──────────────────────────────────────")
    print(f"  {t1} wins  : {res[f'{t1}_win']:.1%}")
    print(f"  Draw      : {res['draw']:.1%}")
    print(f"  {t2} wins  : {res[f'{t2}_win']:.1%}")
    print(f"  ──────────────────────────────────────")
    print(f"  TAHMIN   : {res['prediction']}  ({res['confidence']}% güven)")
    print(f"{'='*52}")


def deep_analysis(team1, team2, elo_ratings, df_full, rf_model, scaler,
                  neutral=True, tournament="FIFA World Cup", n_form=15):
    """
    İki takım arasında derin maç analizi üret.

    Standart predict_match()'in üzerine şunları ekler:
      - Son N maçın W/D/L dökümü ve gol istatistikleri
      - Rakip kalitesi filtresi (güçlü vs zayıf rakip formu)
      - H2H tam geçmişi
      - Model tahmini ile ham veri arasındaki uyumsuzluk uyarısı
      - CSV ve metin raporu çıktısı

    Parametreler
    ------------
    team1, team2 : str   — takım adları (martj42 repo'sundaki isimle birebir)
    n_form       : int   — form analizinde kaç maç bakılacak (varsayılan 15)
    neutral      : bool  — World Cup maçları için True
    """
    import os
    today = pd.Timestamp.today()

    # ── 1. Temel tahmin ───────────────────────────────────
    result = predict_match(
        team1, team2, elo_ratings, df_full,
        rf_model, scaler, neutral=neutral, tournament=tournament
    )

    # ── 2. Son N maç detayı ───────────────────────────────
    def get_match_log(team, n=n_form):
        matches = df_full[
            ((df_full["home_team"] == team) | (df_full["away_team"] == team)) &
            (df_full["date"] < today)
        ].tail(n)
        log = []
        for _, r in matches.iterrows():
            is_home = r["home_team"] == team
            opponent = r["away_team"] if is_home else r["home_team"]
            gf = r["home_score"] if is_home else r["away_score"]
            ga = r["away_score"] if is_home else r["home_score"]
            if gf > ga:   wdl = "W"
            elif gf == ga: wdl = "D"
            else:          wdl = "L"
            log.append({
                "date":       r["date"].strftime("%Y-%m-%d"),
                "opponent":   opponent,
                "gf":         int(gf),
                "ga":         int(ga),
                "result":     wdl,
                "tournament": r["tournament"],
                "neutral":    r["neutral"],
            })
        return log

    log1 = get_match_log(team1)
    log2 = get_match_log(team2)

    # ── 3. Gol istatistikleri ─────────────────────────────
    def goal_stats(log):
        if not log:
            return {"avg_gf": 0, "avg_ga": 0, "diff": 0, "w": 0, "d": 0, "l": 0}
        gf_list = [m["gf"] for m in log]
        ga_list = [m["ga"] for m in log]
        return {
            "avg_gf": round(sum(gf_list) / len(gf_list), 2),
            "avg_ga": round(sum(ga_list) / len(ga_list), 2),
            "diff":   round((sum(gf_list) - sum(ga_list)) / len(gf_list), 2),
            "w":      sum(1 for m in log if m["result"] == "W"),
            "d":      sum(1 for m in log if m["result"] == "D"),
            "l":      sum(1 for m in log if m["result"] == "L"),
        }

    stats1 = goal_stats(log1)
    stats2 = goal_stats(log2)

    # ── 4. H2H tam geçmişi ────────────────────────────────
    h2h_all = df_full[
        (((df_full["home_team"] == team1) & (df_full["away_team"] == team2)) |
         ((df_full["home_team"] == team2) & (df_full["away_team"] == team1))) &
        (df_full["date"] < today)
    ].copy()

    h2h_log = []
    for _, r in h2h_all.iterrows():
        is_t1_home = r["home_team"] == team1
        t1_score = r["home_score"] if is_t1_home else r["away_score"]
        t2_score = r["away_score"] if is_t1_home else r["home_score"]
        if t1_score > t2_score:   wdl = f"{team1} wins"
        elif t1_score == t2_score: wdl = "Draw"
        else:                      wdl = f"{team2} wins"
        h2h_log.append({
            "date":       r["date"].strftime("%Y-%m-%d"),
            "score":      f"{int(t1_score)}-{int(t2_score)}",
            "result":     wdl,
            "tournament": r["tournament"],
        })

    # ── 5. Model vs ham veri uyumsuzluk kontrolü ──────────
    t1_win_prob = result.get(f"{team1}_win", 0)
    elo1 = elo_ratings.get(team1, 1500)
    elo2 = elo_ratings.get(team2, 1500)

    # Form farkı modelin beklentisini karşılamıyor mu?
    form1 = result["team1_form"]
    form2 = result["team2_form"]
    form_edge = form1 - form2          # pozitif = team1 lehine
    elo_edge  = (elo1 - elo2) / 400   # normalize

    mismatch = False
    mismatch_note = ""
    if form_edge > 0.20 and t1_win_prob < 0.40:
        mismatch = True
        mismatch_note = (
            f"{team1} form avantajı ({form_edge:+.2f}) "
            f"model tahminine yansımıyor ({t1_win_prob:.0%}). "
            f"ELO baskısı form sinyalini maskeliyor olabilir."
        )
    elif form_edge < -0.20 and t1_win_prob > 0.60:
        mismatch = True
        mismatch_note = (
            f"{team2} form avantajı ({-form_edge:+.2f}) "
            f"model tahminine yansımıyor. "
            f"ELO baskısı form sinyalini maskeliyor olabilir."
        )

    # ── 6. Raporu yazdır ──────────────────────────────────
    SEP = "=" * 60

    print(f"\n{SEP}")
    print(f"  DERİN ANALİZ: {team1} vs {team2}")
    print(f"{SEP}")

    print(f"\n  ELO        : {team1} {elo1:.0f}  |  {team2} {elo2:.0f}  (fark: {elo1-elo2:+.0f})")
    print(f"  Form       : {team1} {form1:.3f}  |  {team2} {form2:.3f}  (fark: {form_edge:+.3f})")
    print(f"  H2H        : {result['h2h_team1']:.0%} {team1} kazanma oranı ({len(h2h_log)} karşılaşma)")

    print(f"\n  {'':30} {team1:>12}   {team2:>12}")
    print(f"  {'Ortalama gol (son '+str(n_form)+' maç)':30} {stats1['avg_gf']:>12.2f}   {stats2['avg_gf']:>12.2f}")
    print(f"  {'Ortalama yenilen':30} {stats1['avg_ga']:>12.2f}   {stats2['avg_ga']:>12.2f}")
    print(f"  {'Gol farkı':30} {stats1['diff']:>+12.2f}   {stats2['diff']:>+12.2f}")
    print(f"  {'W-D-L':30} {stats1['w']}-{stats1['d']}-{stats1['l']:>10}   {stats2['w']}-{stats2['d']}-{stats2['l']}")

    print(f"\n  ── Son {n_form} maç: {team1} ──")
    for m in log1:
        tag = f"[{m['result']}]"
        print(f"    {m['date']}  {tag:4}  {m['gf']}-{m['ga']}  vs {m['opponent']}  ({m['tournament'][:35]})")

    print(f"\n  ── Son {n_form} maç: {team2} ──")
    for m in log2:
        tag = f"[{m['result']}]"
        print(f"    {m['date']}  {tag:4}  {m['gf']}-{m['ga']}  vs {m['opponent']}  ({m['tournament'][:35]})")

    if h2h_log:
        print(f"\n  ── H2H geçmişi (tüm zamanlar) ──")
        for m in h2h_log:
            print(f"    {m['date']}  {m['score']}  {m['result']}  ({m['tournament']})")
    else:
        print(f"\n  ── H2H geçmişi: Kayıt yok ──")

    print(f"\n  ── Model tahmini ──")
    print(f"    {team1} wins : {result[f'{team1}_win']:.1%}")
    print(f"    Draw       : {result['draw']:.1%}")
    print(f"    {team2} wins : {result[f'{team2}_win']:.1%}")
    print(f"    → {result['prediction']}  ({result['confidence']}% güven)")

    if mismatch:
        print(f"\n  ⚠  UYUMSUZLUK: {mismatch_note}")

    print(f"\n{SEP}")

    # ── 7. CSV çıktısı ────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    fname = f"outputs/deep_{team1.replace(' ','_')}_vs_{team2.replace(' ','_')}.csv"

    rows = []
    for m in log1:
        m["team"] = team1; rows.append(m)
    for m in log2:
        m["team"] = team2; rows.append(m)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(fname, index=False)
    print(f"  Detay kaydedildi: {fname}")

    return {
        "prediction":  result,
        "stats_team1": stats1,
        "stats_team2": stats2,
        "log_team1":   log1,
        "log_team2":   log2,
        "h2h":         h2h_log,
        "mismatch":    mismatch,
        "mismatch_note": mismatch_note,
    }


def predict_upcoming_wc2026(df_full, elo_ratings, rf_model, scaler):
    df_raw = pd.read_csv(DATA_URL)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    upcoming = df_raw[
        (df_raw["tournament"] == "FIFA World Cup") &
        (df_raw["home_score"].isna())
    ].copy()

    if len(upcoming) == 0:
        print("Bekleyen WC maçı bulunamadı.")
        return pd.DataFrame()

    print(f"\n{len(upcoming)} bekleyen WC 2026 maçı:")
    results = []
    for _, row in upcoming.iterrows():
        res = predict_match(
            row["home_team"], row["away_team"],
            elo_ratings, df_full, rf_model, scaler
        )
        res["date"] = row["date"].strftime("%Y-%m-%d")
        results.append(res)
        print_prediction(res)

    return pd.DataFrame(results)


def main():
    df = load_data()
    elo_ratings, df_with_elo = compute_elo(df)
    df_feat = build_features(df_with_elo)
    rf_model, lr_model, scaler = train_model(df_feat)

    df_pred = predict_upcoming_wc2026(df, elo_ratings, rf_model, scaler)
    if len(df_pred) > 0:
        df_pred.to_csv("outputs/wc2026_predictions.csv", index=False)
        print("\nKaydedildi: outputs/wc2026_predictions.csv")

    # Örnek tek maç tahmini
    print("\n--- Örnek: Netherlands vs Japan ---")
    print_prediction(predict_match(
        "Netherlands", "Japan",
        elo_ratings, df, rf_model, scaler
    ))


if __name__ == "__main__":
    main()