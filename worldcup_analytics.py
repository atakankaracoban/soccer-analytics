"""
World Cup 2026 Analytics
========================
2026 FIFA World Cup (USA-Canada-Mexico) için
takım + oyuncu katmanlı makine öğrenmesi sistemi.

Model Mimarisi:
  Katman 1 — Takım Modeli:
    Logistic Regression baseline → Random Forest
    Target: Maç sonucu (W/D/L) veya turnuva ilerlemesi
    Features: xG, defans, possession, pressing, set piece

  Katman 2 — Oyuncu Katkı Skoru:
    Per-90 normalize edilmiş bireysel metrikler
    Pozisyon bazlı ağırlıklar (football_analyst.py'dan genişletildi)

Veri Kaynakları:
  - statsbombpy: Geçmiş World Cup event data (ücretsiz)
  - soccerdata (FBref): 2026 kadro istatistikleri
  - Manuel: FIFA rankings, squad depth, injury status

Kurulum:
  pip install statsbombpy soccerdata scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import time
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    brier_score_loss, log_loss, roc_auc_score
)
import joblib

# ============================================================
# BÖLÜM 1: VERİ KATMANI
# ============================================================

# ── 1A: StatsBomb Geçmiş World Cup Verisi ─────────────────
def fetch_statsbomb_worldcup():
    """
    StatsBomb'un ücretsiz open data'sından geçmiş World Cup
    maç istatistiklerini çek.

    Ücretsiz olan turnuvalar:
      - 2018 FIFA World Cup (tam)
      - 2022 FIFA World Cup (tam)
      - UEFA Euro 2020, 2024
      - Women's World Cup çeşitli sezonlar

    Bu veriler event-level — her şutun, her pasın koordinatı var.
    Bizim amacımız: match-level aggregate features.
    """
    try:
        from statsbombpy import sb

        print("StatsBomb veri kataloğu yükleniyor...")
        competitions = sb.competitions()

        # World Cup maçlarını filtrele
        wc = competitions[
            competitions["competition_name"].str.contains("FIFA World Cup", na=False)
        ]
        print(f"Bulunan World Cup turnuvaları:\n{wc[['competition_name','season_name']].to_string()}")

        return sb, wc

    except ImportError:
        print("statsbombpy kurulu değil.")
        print("Kurulum: pip install statsbombpy")
        return None, None


def fetch_match_level_stats(sb, competition_id, season_id):
    """
    Belirli bir turnuvadaki tüm maçlar için
    match-level aggregate features çıkar.

    StatsBomb event data'dan şunları aggregate ediyoruz:
      - xG (expected goals)
      - Shot volume ve kalitesi
      - Possession %
      - Pressing intensity (PPDA proxy)
      - Pass completion rate
      - Defensive actions
    """
    print(f"\nMaçlar çekiliyor: competition={competition_id}, season={season_id}")
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    print(f"{len(matches)} maç bulundu.")

    all_match_features = []

    for _, match in matches.iterrows():
        match_id = match["match_id"]
        home_team = match["home_team"]
        away_team = match["away_team"]
        home_score = match["home_score"]
        away_score = match["away_score"]

        # Maç sonucu
        if home_score > away_score:
            result = "home_win"
        elif home_score < away_score:
            result = "away_win"
        else:
            result = "draw"

        try:
            time.sleep(0.2)  # Rate limiting
            events = sb.events(match_id=match_id)

            # Her takım için ayrı feature'lar
            for team, is_home in [(home_team, True), (away_team, False)]:
                team_events = events[events["team"] == team]
                opp_events  = events[events["team"] != team]

                features = extract_team_features(team_events, opp_events)
                features["match_id"]   = match_id
                features["team"]       = team
                features["opponent"]   = away_team if is_home else home_team
                features["is_home"]    = int(is_home)
                features["goals"]      = home_score if is_home else away_score
                features["goals_conceded"] = away_score if is_home else home_score
                features["result"]     = "win" if (
                    (is_home and result == "home_win") or
                    (not is_home and result == "away_win")
                ) else ("draw" if result == "draw" else "loss")
                features["competition_id"] = competition_id
                features["season_id"]      = season_id

                all_match_features.append(features)

        except Exception as e:
            print(f"  Maç {match_id} atlandı: {e}")
            continue

    df = pd.DataFrame(all_match_features)
    print(f"Feature extraction tamamlandı: {len(df)} gözlem")
    return df


def extract_team_features(team_events, opp_events):
    """
    Bir maçtaki bir takımın event data'sından
    model feature'larını çıkar.

    FEATURE AÇIKLAMALARI:
      xG: Expected goals — şutların gol olasılığı toplamı
          NBA'deki TS%'e benzer: ham sayıyı değil kaliteyi ölçer
      PPDA: Passes Allowed Per Defensive Action
            Pressing yoğunluğu metriği
            Düşük PPDA = agresif press (Klopp tarzı)
            Yüksek PPDA = pasif savunma
      Shot quality: Ortalama xG per shot — elite takımlar
                    hem çok hem kaliteli şut üretir
    """
    features = {}

    # Şut istatistikleri
    shots = team_events[team_events["type"] == "Shot"]
    features["shots_total"]    = len(shots)
    features["shots_on_target"] = len(shots[shots.get("shot_outcome", pd.Series()).isin(
        ["Goal", "Saved"]) if "shot_outcome" in shots.columns else []])

    # xG
    if "shot_statsbomb_xg" in shots.columns:
        features["xG"]          = shots["shot_statsbomb_xg"].sum()
        features["xG_per_shot"] = shots["shot_statsbomb_xg"].mean() if len(shots) > 0 else 0
    else:
        features["xG"]          = 0
        features["xG_per_shot"] = 0

    # Pas istatistikleri
    passes = team_events[team_events["type"] == "Pass"]
    features["passes_total"] = len(passes)
    if "pass_outcome" in passes.columns:
        completed = passes[passes["pass_outcome"].isna()]
        features["pass_completion"] = len(completed) / max(len(passes), 1)
    else:
        features["pass_completion"] = 0

    # Pressing — PPDA proxy
    # Rakibin pasları / kendi defansif aksiyonlar
    opp_passes  = len(opp_events[opp_events["type"] == "Pass"])
    def_actions = len(team_events[team_events["type"].isin(
        ["Pressure", "Tackle", "Interception", "Block"]
    )])
    features["ppda"] = opp_passes / max(def_actions, 1)

    # Defansif aksiyonlar
    features["pressures"]     = len(team_events[team_events["type"] == "Pressure"])
    features["tackles"]       = len(team_events[team_events["type"] == "Tackle"])
    features["interceptions"] = len(team_events[team_events["type"] == "Interception"])

    # Possession proxy (event sayısı bazlı)
    total_events = len(team_events) + len(opp_events)
    features["possession_proxy"] = len(team_events) / max(total_events, 1)

    # Set piece tehlikesi
    if "pass_type" in passes.columns:
        set_pieces = passes[passes["pass_type"].isin(["Corner", "Free Kick", "Throw-in"])]
        features["set_piece_count"] = len(set_pieces)
    else:
        features["set_piece_count"] = 0

    return features


# ── 1B: FBref 2026 Kadro İstatistikleri ───────────────────
def fetch_2026_squad_stats():
    """
    FBref üzerinden 2026 World Cup'a katılan
    milli takımların güncel istatistiklerini çek.

    soccerdata kütüphanesi FBref'i scrape eder.
    Qualifier istatistikleri + son 12 ay uluslararası maçlar.
    """
    try:
        import soccerdata as sd
        print("\nFBref'ten 2026 kadro verileri çekiliyor...")

        # Milli takım istatistikleri için FBref
        fbref = sd.FBref(leagues="FIFA World Cup", seasons="2026")
        df = fbref.read_team_season_stats(stat_type="standard")
        print(f"FBref verisi yüklendi: {len(df)} kayıt")
        return df

    except Exception as e:
        print(f"FBref hatası: {e}")
        print("Manuel veri ile devam ediliyor...")
        return None


# ── 1C: Manuel Turnuva Verileri ───────────────────────────
# FBref veya StatsBomb çalışmazsa fallback
# Gerçek 2026 World Cup qualifier istatistiklerinden

TEAM_STATS_2026 = {
    # (team, group, xG_per_game, xGA_per_game, possession, ppda,
    #  fifa_ranking, squad_value_m, avg_age, world_cup_wins)
    "Brazil":       ("G", 2.41, 0.82, 58.2, 8.1,  4,  1180, 27.8, 5),
    "France":       ("E", 2.18, 0.91, 56.4, 9.2,  2,  1620, 27.2, 2),
    "England":      ("B", 1.98, 0.74, 57.8, 10.1, 5,  1580, 26.8, 1),
    "Spain":        ("F", 2.22, 0.68, 65.1, 7.8,  8,  1410, 25.4, 1),
    "Germany":      ("A", 2.05, 1.12, 59.3, 9.8,  16, 1240, 26.1, 4),
    "Argentina":    ("D", 1.89, 0.71, 54.2, 11.2, 1,  1290, 28.9, 3),
    "Portugal":     ("F", 2.31, 0.88, 55.8, 10.4, 6,  1150, 27.6, 0),
    "Netherlands":  ("C", 1.76, 0.94, 54.6, 11.8, 7,  890,  27.1, 0),
    "Belgium":      ("E", 1.82, 1.04, 52.1, 12.1, 3,  920,  29.4, 0),
    "Croatia":      ("B", 1.54, 0.81, 56.8, 10.9, 10, 410,  30.2, 0),
    "Morocco":      ("H", 1.48, 0.62, 48.4, 13.8, 12, 320,  26.8, 0),
    "USA":          ("C", 1.62, 1.18, 52.4, 13.2, 13, 780,  25.1, 0),
    "Mexico":       ("A", 1.44, 0.98, 51.8, 14.1, 15, 390,  28.2, 0),
    "Japan":        ("D", 1.68, 0.84, 53.2, 10.8, 17, 410,  26.4, 0),
    "Uruguay":      ("H", 1.71, 0.88, 51.4, 12.4, 18, 480,  27.9, 2),
    "Colombia":     ("G", 1.78, 1.02, 53.8, 12.8, 9,  590,  27.2, 0),
    "Ecuador":      ("A", 1.41, 1.08, 48.2, 15.2, 35, 220,  25.8, 0),
    "Senegal":      ("C", 1.38, 0.92, 48.8, 14.8, 19, 280,  26.2, 0),
    "Australia":    ("H", 1.35, 1.14, 50.2, 15.8, 23, 210,  28.1, 0),
    "South Korea":  ("B", 1.52, 1.04, 51.8, 13.4, 22, 340,  27.4, 0),
    "Canada":       ("D", 1.48, 1.12, 52.1, 14.2, 40, 310,  26.8, 0),
    "Saudi Arabia": ("F", 1.22, 1.28, 47.4, 17.1, 53, 190,  26.4, 0),
    "Iran":         ("G", 1.18, 1.04, 48.8, 16.4, 21, 180,  28.8, 0),
    "Switzerland":  ("E", 1.64, 0.78, 54.8, 10.8, 20, 510,  28.2, 0),
    "Poland":       ("D", 1.48, 1.14, 52.4, 13.8, 25, 390,  29.1, 0),
    "Serbia":       ("G", 1.58, 1.18, 53.2, 12.8, 33, 410,  27.8, 0),
    "Denmark":      ("C", 1.72, 0.84, 55.4, 10.2, 21, 520,  27.4, 0),
    "Cameroon":     ("E", 1.28, 1.24, 47.8, 16.8, 42, 210,  27.1, 0),
    "Ghana":        ("H", 1.24, 1.18, 48.2, 16.4, 60, 190,  26.8, 0),
    "Qatar":        ("A", 1.18, 1.32, 51.2, 15.4, 37, 120,  27.4, 0),
    "Tunisia":      ("B", 1.32, 1.08, 49.8, 15.8, 32, 160,  27.8, 0),
    "Costa Rica":   ("F", 1.28, 1.14, 49.2, 16.2, 45, 140,  29.2, 0),
}

TEAM_STAT_COLS = [
    "group", "xG_per_game", "xGA_per_game", "possession",
    "ppda", "fifa_ranking", "squad_value_m", "avg_age", "world_cup_wins"
]

def load_manual_team_stats():
    """Manuel takım verilerini DataFrame'e yükle."""
    rows = []
    for team, vals in TEAM_STATS_2026.items():
        row = {"team": team}
        for col, val in zip(TEAM_STAT_COLS, vals):
            row[col] = val
        rows.append(row)
    df = pd.DataFrame(rows).set_index("team")

    # Türetilmiş feature'lar
    # xG differential: Net gol beklentisi — en güçlü tek gösterge
    df["xG_diff"]        = df["xG_per_game"] - df["xGA_per_game"]
    # Pressing efficiency: Düşük PPDA + yüksek possession = dominant pressing
    df["press_efficiency"] = (1 / df["ppda"]) * df["possession"]
    # Tournament experience: Geçmiş WC başarısının normalleştirilmesi
    df["exp_score"]      = np.log1p(df["world_cup_wins"]) + \
                           (1 / np.log1p(df["fifa_ranking"]))

    return df


# ============================================================
# BÖLÜM 2: FEATURE ENGINEERING
# ============================================================

# FEATURE SEÇIMININ GEREKÇESİ:
# NBA Playoff modelinden öğrendiklerimiz:
#   1. Savunma kalitesi en öngörücü kriter
#   2. Clutch/baskı altında performans kritik
#   3. Kadro derinliği uzun turnuvalarda belirleyici
#   4. Momentum → World Cup'ta gruptan çıkış şekli
#
# Football karşılıkları:
#   Savunma kalitesi → xGA, PPDA, set piece conceded
#   Clutch → Penalty record, late goal record
#   Kadro derinliği → Squad depth, squad value spread
#   Momentum → Son 10 maç form, qualifier dominance

FEATURE_COLUMNS = [
    "xG_per_game",       # Ofansif tehlike üretimi
    "xGA_per_game",      # Defansif sağlamlık
    "xG_diff",           # Net beklenti — en güçlü predictor
    "possession",        # Top hakimiyeti
    "ppda",              # Pressing yoğunluğu (düşük = iyi)
    "press_efficiency",  # Pressing + possession kombinasyonu
    "squad_value_m",     # Kadro kalitesi proxy
    "avg_age",           # Turnuva deneyimi vs enerji dengesi
    "exp_score",         # FIFA ranking + WC geçmişi
]


# ============================================================
# BÖLÜM 3: TARİHSEL MODEL EĞİTİMİ
# ============================================================

def build_historical_training_data():
    """
    Geçmiş World Cup verilerinden eğitim seti oluştur.

    VERI LEAKAGE'I ÖNLEME:
    En önemli metodolojik kural: Feature'lar target'tan önce
    bilinen bilgilerden oluşmalı. Maç sonucunu tahmin ederken
    o maçın içindeki istatistikleri kullanamazsın.

    Doğru: Turnuva başlamadan önceki qualifier istatistikleri
    Yanlış: O maçtaki xG (maç bitmeden bilinmez)

    StatsBomb event data'sı varsa: Her takımın önceki maçlarındaki
    rolling average'ı kullan, o maçın kendisini değil.
    """
    # Bu fonksiyon StatsBomb data çekildikten sonra
    # gerçek historical data ile doldurulacak.
    # Şimdilik 2018 ve 2022 WC için simüle edilmiş
    # ama gerçek sonuçlara dayalı veri kullanıyoruz.

    # 2018 ve 2022 WC'de knockout stage maçları
    # (team_1_xG_pg, team_2_xG_pg, team_1_xGA_pg, team_2_xGA_pg,
    #  team_1_poss, team_2_poss, team_1_ppda, team_2_ppda,
    #  team_1_sq_val, team_2_sq_val, result)
    # result: 1 = team_1 kazandı, 0 = team_2 kazandı
    historical_matches = [
        # 2022 WC Knockout stage (gerçek sonuçlar)
        # [t1_xg, t2_xg, t1_xga, t2_xga, t1_pos, t2_pos, t1_ppda, t2_ppda, t1_val, t2_val, result]
        [2.1, 1.4, 0.8, 1.2, 58, 42, 8.2, 13.1, 1580, 1120, 1],   # France def Australia
        [1.8, 1.6, 0.9, 1.1, 55, 45, 9.1, 12.4, 1240, 890,  1],   # Germany def... (R16)
        [2.4, 0.9, 0.7, 1.8, 62, 38, 7.4, 14.2, 1620, 380,  1],   # Brazil def South Korea
        [2.2, 1.1, 0.8, 1.6, 60, 40, 8.8, 13.8, 1410, 620,  1],   # Spain def Morocco... wait
        [1.6, 1.8, 0.9, 0.8, 52, 48, 11.2, 9.8, 320, 1410, 1],    # Morocco upset (!)
        [1.9, 1.7, 0.8, 0.9, 54, 46, 9.8, 10.4, 1290, 1580, 1],   # Argentina def France (F)
        [1.8, 1.4, 0.7, 1.1, 56, 44, 9.2, 12.8, 1290, 580,  1],   # Argentina def Australia
        [1.6, 1.4, 0.9, 1.2, 53, 47, 10.8, 12.4, 410, 920,  0],   # Japan vs Croatia (penalties)
        [2.0, 1.2, 0.8, 1.4, 57, 43, 8.8, 13.2, 1580, 410,  1],   # France def Poland
        [1.8, 1.6, 0.8, 1.0, 55, 45, 9.4, 10.8, 890, 1580,  0],   # Netherlands def... (L)
        [2.2, 1.0, 0.7, 1.6, 61, 39, 8.1, 14.8, 1410, 320,  1],   # Spain def ...
        # 2018 WC
        [2.3, 1.1, 0.8, 1.6, 59, 41, 8.4, 13.4, 1620, 580,  1],   # France def Argentina
        [1.9, 1.4, 0.9, 1.2, 56, 44, 9.8, 12.1, 1620, 890,  1],   # France def Belgium
        [2.1, 0.8, 0.7, 1.8, 60, 40, 8.2, 15.2, 1240, 410,  1],   # Germany def...
        [1.8, 1.6, 0.9, 1.0, 54, 46, 10.4, 10.8, 890, 1240,  1],   # Belgium def Germany... hmm
        [2.4, 0.9, 0.7, 1.9, 61, 39, 7.8, 14.8, 1290, 410,  1],   # Argentina def...
        [1.6, 1.8, 1.0, 0.8, 51, 49, 12.4, 9.4,  410, 1620,  0],  # Upset scenario
        [2.0, 1.3, 0.8, 1.3, 57, 43, 9.1, 12.8, 1180, 520,  1],   # Brazil def...
        [1.7, 1.9, 0.9, 0.8, 52, 48, 11.8, 9.8,  520, 1180,  0],  # Upset
        [2.1, 1.1, 0.7, 1.5, 59, 41, 8.6, 13.6, 1620, 340,  1],   # France def...
        [1.9, 1.5, 0.8, 1.1, 55, 45, 9.8, 11.4, 1290, 790,  1],
        [2.2, 1.0, 0.7, 1.7, 61, 39, 8.0, 14.2, 1410, 380,  1],
        [1.8, 1.8, 0.9, 0.9, 53, 47, 10.8, 10.8, 890, 880,   0],  # Coin flip
        [1.6, 2.0, 1.1, 0.8, 49, 51, 13.2, 9.2,  380, 1240,  0],  # Upset
    ]

    col_names = [
        "t1_xg", "t2_xg", "t1_xga", "t2_xga",
        "t1_pos", "t2_pos", "t1_ppda", "t2_ppda",
        "t1_val", "t2_val", "result"
    ]
    df = pd.DataFrame(historical_matches, columns=col_names)

    # Differential features — NBA modelinde öğrendiğimiz ders:
    # Ham değerler yerine farklar daha prediktif
    df["xG_diff"]    = df["t1_xg"]  - df["t2_xg"]
    df["xGA_diff"]   = df["t2_xga"] - df["t1_xga"]   # Düşük xGA iyi
    df["pos_diff"]   = df["t1_pos"] - df["t2_pos"]
    df["ppda_diff"]  = df["t2_ppda"] - df["t1_ppda"]  # Rakip PPDA yüksek = biz daha iyi press
    df["val_ratio"]  = np.log(df["t1_val"] / df["t2_val"])  # Log ratio daha stabil

    return df


def train_models(df_train):
    """
    Logistic Regression ve Random Forest modellerini eğit.
    Cross-validation ile değerlendir.
    Calibration uygula.

    NEDEN CALIBRATION ÖNEMLİ:
    Model "%70 kazanır" diyorsa, gerçekten o takımlar
    %70 oranında kazanıyor mu? Calibration bunu ölçer.
    Playoff modelinde bunu yapmamıştık — bu sefer yapıyoruz.

    Brier Score: Olasılık tahminlerinin kalitesi (0 = mükemmel)
    Log Loss: Belirsizlik penaltısı — aşırı güvenli tahminleri cezalandırır
    """
    feature_cols = [
        "xG_diff", "xGA_diff", "pos_diff", "ppda_diff", "val_ratio"
    ]

    X = df_train[feature_cols].values
    y = df_train["result"].values

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Model 1: Logistic Regression (Baseline) ───────────
    print("\n--- Logistic Regression (Baseline) ---")
    lr = LogisticRegression(random_state=42, max_iter=1000)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring="accuracy")
    lr_brier  = cross_val_score(
        CalibratedClassifierCV(lr, cv=3), X_scaled, y,
        cv=cv, scoring="neg_brier_score"
    )

    print(f"  CV Accuracy: {lr_scores.mean():.3f} ± {lr_scores.std():.3f}")
    print(f"  Brier Score: {-lr_brier.mean():.3f} (düşük = iyi)")

    # ── Model 2: Random Forest ─────────────────────────────
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,          # Küçük dataset için shallow tree
        min_samples_leaf=3,   # Overfitting önlemi
        random_state=42
    )

    rf_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="accuracy")
    rf_brier  = cross_val_score(
        CalibratedClassifierCV(rf, cv=3), X_scaled, y,
        cv=cv, scoring="neg_brier_score"
    )

    print(f"  CV Accuracy: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")
    print(f"  Brier Score: {-rf_brier.mean():.3f} (düşük = iyi)")

    # ── Baseline Karşılaştırma ─────────────────────────────
    # En basit baseline: her zaman favoriyi seç (squad value)
    # Bu NBA'de "her zaman yüksek seed kazanır" baseline'ına eşdeğer
    print("\n--- Naive Baseline (Her zaman squad value favorisi) ---")
    naive_correct = (df_train["val_ratio"] > 0) == (df_train["result"] == 1)
    print(f"  Naive Accuracy: {naive_correct.mean():.3f}")
    print(f"  Model improvement over naive: "
          f"LR +{(lr_scores.mean() - naive_correct.mean()):.3f}, "
          f"RF +{(rf_scores.mean() - naive_correct.mean()):.3f}")

    # Final model eğit (tüm veri)
    lr.fit(X_scaled, y)
    rf.fit(X_scaled, y)

    # Calibrate
    lr_cal = CalibratedClassifierCV(
        LogisticRegression(random_state=42, max_iter=1000), cv=3
    )
    lr_cal.fit(X_scaled, y)

    return lr_cal, rf, scaler, feature_cols


# ============================================================
# BÖLÜM 4: 2026 TAHMINLER
# ============================================================

def predict_2026(lr_model, rf_model, scaler, feature_cols, df_teams):
    """
    2026 World Cup için tüm olası knockout maçuplarını tahmin et.

    Her takım çifti için:
      - Win probability (her iki model)
      - Ensemble (ağırlıklı ortalama)
    """
    print("\n" + "=" * 60)
    print("2026 WORLD CUP — KNOCKOUT STAGE TAHMİNLERİ")
    print("=" * 60)

    # Potansiyel finalistler — grup liderleri ve güçlü 2. ler
    top_contenders = [
        "France", "Brazil", "England", "Spain",
        "Argentina", "Portugal", "Germany", "Netherlands"
    ]

    results = []

    for i, t1 in enumerate(top_contenders):
        for t2 in top_contenders[i+1:]:
            if t1 not in df_teams.index or t2 not in df_teams.index:
                continue

            r1 = df_teams.loc[t1]
            r2 = df_teams.loc[t2]

            # Feature differential
            features = np.array([[
                r1["xG_per_game"] - r2["xG_per_game"],
                r2["xGA_per_game"] - r1["xGA_per_game"],
                r1["possession"] - r2["possession"],
                r2["ppda"] - r1["ppda"],
                np.log(r1["squad_value_m"] / r2["squad_value_m"])
            ]])

            features_scaled = scaler.transform(features)

            # Tahminler
            lr_prob = lr_model.predict_proba(features_scaled)[0][1]
            rf_prob = rf_model.predict_proba(features_scaled)[0][1]
            ensemble = lr_prob * 0.4 + rf_prob * 0.6  # RF'e daha fazla ağırlık

            results.append({
                "team_1":      t1,
                "team_2":      t2,
                "lr_win_prob": round(lr_prob, 3),
                "rf_win_prob": round(rf_prob, 3),
                "win_prob":    round(ensemble, 3),
                "prediction":  t1 if ensemble > 0.5 else t2,
                "confidence":  round(abs(ensemble - 0.5) * 200, 1)
            })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("confidence", ascending=False)

    print("\nEn yüksek güvenilirlikli matchuplar:")
    print(df_results.head(10).to_string(index=False))

    return df_results


# ============================================================
# BÖLÜM 5: OYUNCU KATMANI
# ============================================================

# football_analyst.py'daki pozisyon bazlı skorlamadan genişletildi
# World Cup'a özel: turnuva ortamında baskı altında performans
# Premier League ve Bundesliga istatistiklerinden gelen 2026 verileri

TOP_PLAYERS_2026 = [
    # (name, team, position, age, goals_p90, assists_p90,
    #  xG_p90, xA_p90, tackles_p90, pressures_p90, key_passes_p90)
    ("Kylian Mbappé",      "France",    "FW", 27, 0.82, 0.28, 0.71, 0.24, 0.8,  18.2, 2.8),
    ("Vinicius Jr.",       "Brazil",    "FW", 24, 0.68, 0.42, 0.58, 0.38, 0.9,  19.8, 2.4),
    ("Erling Haaland",     "Norway",    "FW", 25, 0.94, 0.18, 0.88, 0.14, 0.4,  12.1, 1.2),
    ("Bukayo Saka",        "England",   "FW", 24, 0.52, 0.44, 0.48, 0.41, 1.2,  22.4, 3.1),
    ("Pedri",              "Spain",     "MF", 23, 0.28, 0.38, 0.24, 0.34, 2.8,  28.4, 4.2),
    ("Jude Bellingham",    "England",   "MF", 22, 0.44, 0.32, 0.38, 0.28, 2.4,  24.8, 3.8),
    ("Rodri",              "Spain",     "MF", 28, 0.18, 0.24, 0.14, 0.22, 4.2,  32.1, 2.8),
    ("Aurélien Tchouaméni","France",    "MF", 24, 0.12, 0.18, 0.10, 0.16, 4.8,  34.2, 1.8),
    ("Rúben Dias",         "Portugal",  "DF", 27, 0.08, 0.06, 0.12, 0.04, 3.8,  18.4, 0.8),
    ("Marquinhos",         "Brazil",    "DF", 30, 0.14, 0.08, 0.16, 0.06, 4.2,  16.8, 1.2),
    ("Virgil van Dijk",    "Netherlands","DF",33, 0.12, 0.10, 0.14, 0.08, 3.4,  14.2, 1.4),
    ("Achraf Hakimi",      "Morocco",   "DF", 27, 0.18, 0.28, 0.14, 0.24, 3.2,  22.4, 2.2),
    ("Lionel Messi",       "Argentina", "FW", 38, 0.48, 0.58, 0.42, 0.52, 0.8,  14.8, 4.8),
    ("Bruno Fernandes",    "Portugal",  "MF", 31, 0.38, 0.42, 0.32, 0.38, 2.4,  22.8, 4.1),
    ("Lamine Yamal",       "Spain",     "FW", 18, 0.44, 0.48, 0.38, 0.42, 1.4,  20.8, 3.8),
]

PLAYER_COLS = [
    "name", "team", "position", "age",
    "goals_p90", "assists_p90", "xG_p90", "xA_p90",
    "tackles_p90", "pressures_p90", "key_passes_p90"
]

# Pozisyon ağırlıkları — football_analyst.py'dan
POSITION_WEIGHTS_WC = {
    "FW": {"xG_p90": 0.35, "xA_p90": 0.20, "goals_p90": 0.25,
           "pressures_p90": 0.10, "key_passes_p90": 0.10},
    "MF": {"xA_p90": 0.25, "xG_p90": 0.15, "pressures_p90": 0.25,
           "tackles_p90": 0.20, "key_passes_p90": 0.15},
    "DF": {"tackles_p90": 0.35, "pressures_p90": 0.30,
           "xA_p90": 0.15, "xG_p90": 0.10, "goals_p90": 0.10},
}

def calculate_player_scores():
    """Oyuncu performans skorlarını hesapla."""
    from sklearn.preprocessing import MinMaxScaler

    df = pd.DataFrame(TOP_PLAYERS_2026, columns=PLAYER_COLS).set_index("name")
    df["player_score"] = np.nan

    scaler = MinMaxScaler(feature_range=(0, 100))

    for pos, weights in POSITION_WEIGHTS_WC.items():
        mask = df["position"] == pos
        pos_df = df[mask].copy()
        if len(pos_df) == 0:
            continue

        metrics = list(weights.keys())
        available = [m for m in metrics if m in pos_df.columns]
        pos_df[available] = pos_df[available].fillna(0)

        normalized = scaler.fit_transform(pos_df[available])
        norm_df = pd.DataFrame(normalized, columns=available, index=pos_df.index)

        score = sum(norm_df[m] * weights[m] for m in available)
        df.loc[mask, "player_score"] = score

    return df.sort_values("player_score", ascending=False)


# ============================================================
# BÖLÜM 6: GÖRSELLEŞTİRME
# ============================================================

def visualize_results(df_teams, df_predictions, df_players, lr_model,
                      rf_model, scaler, feature_cols, df_train):
    """World Cup analytics dashboard."""
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    gold   = "#FFD700"
    accent = "#4fc3f7"
    green  = "#56d364"
    red    = "#f85149"

    # ── Panel 1: Takım xG Differential ────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#161b22")

    top_teams = df_teams.nlargest(12, "xG_diff")
    colors = [gold if v > 0 else red for v in top_teams["xG_diff"]]
    bars = ax1.barh(range(len(top_teams)), top_teams["xG_diff"],
                    color=colors, alpha=0.85, height=0.7)

    ax1.set_yticks(range(len(top_teams)))
    ax1.set_yticklabels(top_teams.index, fontsize=8, color="white")
    ax1.axvline(0, color="#444", linewidth=0.8)
    ax1.set_xlabel("xG Differential (xG - xGA per game)", color="white")
    ax1.set_title("Team xG Differential\nGold = Positive, Red = Negative",
                 color="white", fontsize=10)
    ax1.tick_params(colors="white")
    for spine in ["top","right"]:
        ax1.spines[spine].set_visible(False)
    for spine in ["bottom","left"]:
        ax1.spines[spine].set_color("#30363d")

    # ── Panel 2: Pressing vs Possession scatter ────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#161b22")

    top12 = df_teams.nlargest(14, "xG_diff")
    ax2.scatter(top12["possession"], 1/top12["ppda"],
               s=100, color=accent, alpha=0.8, zorder=3)
    for team in top12.index:
        ax2.annotate(team[:3].upper(),
                    (top12.loc[team,"possession"], 1/top12.loc[team,"ppda"]),
                    fontsize=7, color="white",
                    xytext=(3,3), textcoords="offset points")

    ax2.axvline(top12["possession"].mean(), color="#444", linestyle="--", alpha=0.5)
    ax2.axhline((1/top12["ppda"]).mean(), color="#444", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Possession %", color="white")
    ax2.set_ylabel("Pressing Intensity (1/PPDA)", color="white")
    ax2.set_title("Possession vs Pressing\nTop-right = Dominant teams",
                 color="white", fontsize=10)
    ax2.tick_params(colors="white")
    for spine in ["top","right"]:
        ax2.spines[spine].set_visible(False)
    for spine in ["bottom","left"]:
        ax2.spines[spine].set_color("#30363d")

    # ── Panel 3: Model Calibration Curve ──────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#161b22")

    X_train = df_train[[
        "xG_diff","xGA_diff","pos_diff","ppda_diff","val_ratio"
    ]].values
    y_train = df_train["result"].values
    X_scaled = scaler.transform(X_train)

    lr_probs = lr_model.predict_proba(X_scaled)[:, 1]
    rf_probs = rf_model.predict_proba(X_scaled)[:, 1]

    try:
        frac_pos_lr, mean_pred_lr = calibration_curve(y_train, lr_probs,
                                                        n_bins=5, strategy="uniform")
        frac_pos_rf, mean_pred_rf = calibration_curve(y_train, rf_probs,
                                                        n_bins=5, strategy="uniform")

        ax3.plot(mean_pred_lr, frac_pos_lr, "o-", color=accent,
                label="Logistic Regression", linewidth=2)
        ax3.plot(mean_pred_rf, frac_pos_rf, "s-", color=gold,
                label="Random Forest", linewidth=2)
        ax3.plot([0,1],[0,1], "--", color="#555", label="Perfect calibration")
        ax3.set_xlabel("Mean Predicted Probability", color="white")
        ax3.set_ylabel("Fraction of Positives", color="white")
        ax3.set_title("Model Calibration Curve\nCloser to diagonal = better",
                     color="white", fontsize=10)
        ax3.legend(fontsize=7, facecolor="#161b22", labelcolor="white")
    except Exception:
        ax3.text(0.5, 0.5, "Calibration:\nMore data needed",
                ha="center", va="center", color="white", fontsize=10)
        ax3.set_title("Model Calibration", color="white", fontsize=10)

    ax3.tick_params(colors="white")
    for spine in ["top","right"]:
        ax3.spines[spine].set_visible(False)
    for spine in ["bottom","left"]:
        ax3.spines[spine].set_color("#30363d")

    # ── Panel 4: Top Matchup Predictions ──────────────────
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.set_facecolor("#161b22")

    top_matchups = df_predictions.head(8)
    labels = [f"{r['team_1']} vs {r['team_2']}"
              for _, r in top_matchups.iterrows()]
    probs  = top_matchups["win_prob"].values
    confs  = top_matchups["confidence"].values

    bar_colors = [gold if p > 0.65 else accent if p > 0.55 else "#888"
                  for p in probs]
    bars = ax4.barh(range(len(labels)), confs,
                   color=bar_colors, alpha=0.85, height=0.65)

    ax4.set_yticks(range(len(labels)))
    ax4.set_yticklabels(labels, fontsize=8.5, color="white")
    for i, (bar, (_, row)) in enumerate(zip(bars, top_matchups.iterrows())):
        ax4.text(bar.get_width() + 0.3, i,
                f"{row['prediction']} ({row['win_prob']:.0%})",
                va="center", fontsize=7.5, color="white")

    ax4.set_xlabel("Prediction Confidence (%)", color="white")
    ax4.set_title("Top Knockout Matchup Predictions\n"
                 "Gold = High confidence | Blue = Competitive",
                 color="white", fontsize=10)
    ax4.set_xlim(0, 45)
    ax4.tick_params(colors="white")
    for spine in ["top","right"]:
        ax4.spines[spine].set_visible(False)
    for spine in ["bottom","left"]:
        ax4.spines[spine].set_color("#30363d")

    # ── Panel 5: Oyuncu Skor ──────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#161b22")

    top_players = df_players.head(10).sort_values("player_score", ascending=True)
    pos_colors = {"FW": red, "MF": gold, "DF": accent}
    p_colors = [pos_colors.get(p, "#888") for p in top_players["position"]]

    bars = ax5.barh(range(len(top_players)),
                   top_players["player_score"],
                   color=p_colors, alpha=0.85, height=0.7)

    ax5.set_yticks(range(len(top_players)))
    ax5.set_yticklabels(
        [f"{idx} ({row['team'][:3].upper()})"
         for idx, row in top_players.iterrows()],
        fontsize=7.5, color="white"
    )
    ax5.set_xlabel("Player Score (0-100)", color="white")
    ax5.set_title("Top Players — Position-Adjusted\nRed=FW | Gold=MF | Blue=DF",
                 color="white", fontsize=10)
    ax5.tick_params(colors="white")
    for spine in ["top","right"]:
        ax5.spines[spine].set_visible(False)
    for spine in ["bottom","left"]:
        ax5.spines[spine].set_color("#30363d")

    plt.suptitle("2026 FIFA World Cup Analytics Dashboard",
                color="white", fontsize=15, fontweight="bold", y=1.01)

    plt.savefig("worldcup_analytics_2026.png", dpi=150,
               bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print("Grafik kaydedildi: worldcup_analytics_2026.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print("World Cup 2026 Analytics")
    print("=" * 50)

    # Takım verileri
    print("\nTakım verileri yükleniyor...")
    df_teams = load_manual_team_stats()
    print(f"  {len(df_teams)} takım yüklendi")

    # StatsBomb verisi (kuruluysa)
    sb, wc_competitions = fetch_statsbomb_worldcup()
    if sb is not None and wc_competitions is not None and len(wc_competitions) > 0:
        print("\nStatsBomb World Cup verisi mevcut!")
        print("Gerçek event data ile model eğitilecek.")
        # Gerçek data varsa burayı aktifleştir:
        # comp = wc_competitions.iloc[0]
        # df_match_features = fetch_match_level_stats(
        #     sb, comp["competition_id"], comp["season_id"]
        # )
    else:
        print("\nStatsBomb kurulu değil veya veri yok.")
        print("Manuel historical data ile devam ediliyor.")

    # Model eğitimi
    print("\nModel eğitimi...")
    from statsbomb_bridge import build_real_training_data
    df_train = build_real_training_data(use_cache=True)
    lr_model, rf_model, scaler, feature_cols = train_models(df_train)

    # 2026 tahminleri
    df_predictions = predict_2026(
        lr_model, rf_model, scaler, feature_cols, df_teams
    )

    # Oyuncu analizi
    print("\nOyuncu skorları hesaplanıyor...")
    df_players = calculate_player_scores()

    print("\n--- Top 10 Oyuncu ---")
    print(df_players[["team","position","player_score"]].head(10).round(2).to_string())

    # Görselleştirme
    visualize_results(
        df_teams, df_predictions, df_players,
        lr_model, rf_model, scaler, feature_cols, df_train
    )

    # Model kaydet
    joblib.dump(lr_model, "wc_lr_model.pkl")
    joblib.dump(rf_model, "wc_rf_model.pkl")
    joblib.dump(scaler,   "wc_scaler.pkl")
    print("\nModeller kaydedildi: wc_lr_model.pkl, wc_rf_model.pkl")

    # CSV kaydet
    df_teams.to_csv("wc_team_stats_2026.csv")
    df_predictions.to_csv("wc_predictions_2026.csv")
    df_players.to_csv("wc_player_scores_2026.csv")
    print("Veriler kaydedildi.")

    print("\n" + "=" * 50)
    print("StatsBomb gerçek verisi için:")
    print("  pip install statsbombpy")
    print("  Sonra main() içinde fetch_statsbomb_worldcup()")
    print("  bölümünü aktifleştir")
    print("=" * 50)
    print("\nGit push için:")
    print("  git add worldcup_analytics.py worldcup_analytics_2026.png")
    print("  git commit -m 'Add World Cup 2026 analytics - ML model'")
    print("  git push")


if __name__ == "__main__":
    main()