# soccer-analytics
# Soccer Analytics — Premier League Player Performance

A football analytics toolkit built as a companion to the NBA Analytics project.
Applies the same analytical framework across a different sport.

## What This Does

- **Position-based performance scoring** — separate weight systems for FW, MF, DF
- **xG analysis** — expected goals vs actual goals (who overperforms? who underperforms?)
- **Per 90 normalization** — the football equivalent of per-game stats
- **xG overperformance detection** — identifying elite finishers vs lucky ones

## Key Concepts

| NBA | Football |
|-----|----------|
| Per Game | Per 90 minutes |
| Points | Goals + xG |
| Assists | Assists + xA |
| STL/BLK | Tackles / Interceptions / Pressures |
| Box score | FBref / StatsBomb event data |

## Files

- `football_analyst.py` — Main toolkit
- `football_analysis.png` — xG scatter + performance score chart

## Notable Findings (2024-25 PL Season)

- **Haaland** near-zero xG overperformance (−0.1): not because he's a poor finisher,
  but because he consistently creates *already high-xG chances* — no need to overperform
- **Gabriel Magalhães** top scorer in position-adjusted ranking: dominant in tackles,
  interceptions, and pressing within the DF category
- **Chris Wood** highest overperformance (+2.8): Nottm Forest's surprise season

## Data Source

Sample data: real 2024-25 FBref statistics (manually compiled)  
Live data: `soccerdata` library → FBref scraper  

```python
import soccerdata as sd
fbref = sd.FBref(leagues="ENG-Premier League", seasons="2024-2025")
df = fbref.read_player_season_stats(stat_type="standard")
```

## Part of a Larger Portfolio

→ [NBA Analytics Toolkit](../README.md) — the original project this methodology comes from
