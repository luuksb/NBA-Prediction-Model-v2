# CLAUDE.md — NBA Playoff Prediction Model

## Project Overview

Monte Carlo bracket simulation model for NBA playoff outcomes. Series-level logit model predicts P(higher seed wins) for each matchup; 50,000 bracket iterations per year produce championship probabilities and round-by-round advancement rates.

Three training windows: 1980–2024, 2000–2024, 2014–2024. Validation year: 2025. Prediction year: 2026.

## Architecture Principles

### Module Independence
- Five modules: `data`, `model`, `injury`, `simulation`, `dashboard`
- Modules communicate **only** through files in `data/` and `results/` directories
- **No cross-module imports.** Never import from `src.model` inside `src.simulation`, etc.
- Each module has a standalone CLI entry point in `scripts/run_*.py`
- The chosen model is stored as a JSON config (feature list + coefficients + window), not a pickled object

### Configuration-Driven
- Feature definitions live in `configs/features.yaml` — no hardcoded feature names in code
- Training windows defined in `configs/training_windows.yaml` — no hardcoded years
- Model selection parameters in `configs/model_selection.yaml`
- If you're tempted to hardcode a feature name, year, or threshold: put it in config instead

### Data Contracts
- `data/final/series_dataset.parquet`: one row per historical playoff series. Columns: all candidate features + metadata (year, round, team_high, team_low, series_id, actual_winner)
- `results/model_selection/chosen_model_{window}.json`: locked model spec per training window (feature list, coefficients, intercept, n_obs). Windows: `full`, `modern`, `recent`
- `results/simulations/{year}_{window}/`: bracket simulation outputs
- `results/injury_sims/injury_sims_{year}.parquet`: injury availability distributions for out-of-sample years

## Data Sources (Priority Order)

1. **Static CSVs from Kaggle** — primary source, placed in `data/raw/`
2. **`nba_api`** — for supplemental data not in Kaggle CSVs
3. **Basketball Reference scraping** — try exactly once if needed. If it blocks (403, rate limit, CAPTCHA), **never attempt again**. Log the failure and move on.

## Feature Registry

All features defined in `configs/features.yaml`. Features are expressed as **deltas** (high seed minus low seed) in the model. Full registry is in config; highlights below.

**Team-level** (produced by `src/data/steps/team_ratings.py`):
- Offensive / defensive / net rating per 100 possessions
- Effective field goal %, true shooting %, three-point attempt rate
- Pace, total rebound rate, offensive rebound rate, turnover rate
- Roster average age (minutes-weighted)

**Playoff experience** (produced by `src/data/steps/playoff_experience.py`):
- Sum / average of historical playoff series wins and series played (up to but not including current year)

**Player-level** (produced by `src/data/steps/player_ratings.py`):
- Top-3 players per team by configurable composite rating (BPM, RAPTOR, PER, WS/48, usage)
- Availability-weighted BPM sum (`bpm_avail_sum`) — **one of the four chosen model features**
- Star flag (`star_flag`) — binary indicator for presence of a top-tier star player — **one of the four chosen model features**

**Coach-level** (produced by `src/data/steps/coach_experience.py`):
- Playoff series win percentage (up to but not including current year)

**Matchup-level:**
- Home-court advantage dummy

**Injury-derived** (Module 3 output, out-of-sample years only; produced by `src/data/steps/player_availability.py`):
- Weighted team availability percentage (`availability_pct`) — series-level

When extending features: add to `configs/features.yaml` with name, type, description, producing step, and active flag. Then implement the producing step in `src/data/steps/`.

## Chosen Model (Locked)

Three window variants locked in `results/model_selection/`:

| Window | File | n_obs | Features |
|--------|------|-------|---------|
| full (1980–2024) | `chosen_model_full.json` | 659 | `delta_bpm_avail_sum`, `delta_playoff_series_wins`, `delta_ts_percent` |
| modern (2000–2024) | `chosen_model_modern.json` | 375 | same |
| recent (2014–2024) | `chosen_model_recent.json` | 165 | same |

All three windows converged on the same 3-feature set. `delta_star_flag` was removed — insignificant in all windows (p=0.24/0.18/0.32) and dropping it improved BIC by ~4–5 points per window. Model selection used BIC with combo sizes 2–5; forbidden pairs prevent collinear combinations (see `configs/model_selection.yaml`).

## Key Design Decisions

- **No play-in tournament.** Bracket starts from 16 playoff teams for all years including 2021+
- **Injury simulation** estimates percentage of games played per series directly (not per-game then aggregated)
- **Historical years** in training data use actual games-played percentages; injury simulation only generates values for 2025/2026
- **Top 3 players** per team identified by configurable weighting of BPM/RAPTOR (set in config)
- **Anti-look-ahead constraint:** all features must be computable using only data available before the series starts. Coach win %, player availability history, team stats — all "up to but not including" the target series/year

## Repository Structure

```
nba-playoff-model/
├── CLAUDE.md
├── pyproject.toml
├── configs/
│   ├── features.yaml
│   ├── training_windows.yaml
│   └── model_selection.yaml
├── data/
│   ├── raw/                   # Immutable, gitignored
│   ├── intermediate/          # Per-step outputs
│   ├── final/                 # series_dataset.parquet
│   └── quality_reports/       # DQ outputs
├── results/
│   ├── model_selection/
│   ├── simulations/
│   └── injury_sims/
├── src/
│   ├── data/
│   │   ├── fetch.py
│   │   ├── steps/
│   │   │   ├── team_ratings.py        # Team offensive/defensive/net ratings, pace, etc.
│   │   │   ├── player_ratings.py      # Top-3 player composite ratings + availability weighting
│   │   │   ├── player_availability.py # Historical games-played % per player per series
│   │   │   ├── playoff_experience.py  # Cumulative series wins/played per team
│   │   │   └── coach_experience.py    # Coach playoff series win %
│   │   ├── assemble.py
│   │   └── quality.py
│   ├── model/
│   │   ├── feature_sets.py
│   │   ├── fit.py
│   │   ├── evaluate.py
│   │   ├── benchmark.py
│   │   └── select.py
│   ├── injury/
│   │   ├── identify_top_players.py
│   │   ├── availability_history.py
│   │   ├── simulate.py
│   │   └── export.py
│   ├── simulation/
│   │   ├── bracket.py
│   │   ├── simulate_series.py
│   │   ├── run_bracket.py
│   │   ├── aggregate.py
│   │   └── report.py
│   └── dashboard/
├── tests/                     # Mirrors src/ structure
├── scripts/
│   ├── run_data_pipeline.py
│   ├── run_model_selection.py
│   ├── run_injury_sim.py
│   ├── run_bracket_sim.py
│   ├── run_dashboard.py
│   └── analysis/              # Off-pipeline analysis scripts (not part of the build order)
│       └── compare_models_insample.py  # LM vs LM+SF in-sample comparison (cross-module imports intentional)
└── notebooks/                 # Exploration only, never production code
```

## Python Environment

- **Python version:** 3.9.13
- **Executable:** `C:/Users/luuks/AppData/Local/Programs/Python/Python39/python.exe`
- Always use the full path above in bash commands (e.g. `C:/Users/luuks/AppData/Local/Programs/Python/Python39/python.exe scripts/run_data_pipeline.py`)
- Do **not** rely on `python`, `python3`, or `py` aliases — they may not resolve correctly in the bash shell on this machine

## Coding Standards

### Style
- Python 3.9+
- Type hints on all function signatures
- Docstrings on all public functions (Google style)
- `ruff` for linting and formatting
- No wildcard imports

### Data Processing
- Pandas for DataFrames; each preprocessing step is a pure function: `DataFrame → DataFrame`
- Parquet for all intermediate and final data files (not CSV)
- Never mutate DataFrames in-place; always return new ones

### Testing
- `pytest` for all tests
- Unit test every function; integration test every module pipeline
- Tests use small fixture datasets, not real data
- Test file naming: `tests/data/test_fetch.py`, `tests/model/test_fit.py`, etc.
- Each function + its test = one commit

### Git Workflow
- One branch per module: `feature/data-pipeline`, `feature/model-selection`, etc.
- Small atomic commits (one step, one function, one test)
- Module-level integration test passes before merging to main
- Review checklist before merge: data contract adherence, no cross-module imports, test coverage, docstrings, config-driven behavior

## Build Order

1. **Data pipeline** (`src/data/`) — produces `series_dataset.parquet`
2. **Model selection** (`src/model/`) — depends on dataset
3. **Injury simulation** (`src/injury/`) — architecturally independent, can parallel with #2
4. **Bracket simulation** (`src/simulation/`) — depends on chosen model + optionally injury outputs
5. **Dashboard** (`src/dashboard/`) — reads result files, built last

## Common Pitfalls to Avoid

- Don't use future data when computing features — everything must be "as of before this series"
- Don't pickle models — use the JSON spec so simulation stays decoupled
- Don't put production logic in notebooks
- Don't import across module boundaries — if you need shared utilities, that's a sign the data contract needs updating. Exception: `scripts/analysis/` scripts may cross-import for one-off research; they are never part of the build pipeline
- Don't hardcode 2025 or 2026 — use config values so years are easily updatable
- Don't retry Basketball Reference if it blocks once — log and use alternative sources

## Available Review Skills
- `structural-review/SKILL.md` — architecture, module boundaries, SoC
- `style-review/SKILL.md` — naming, docs, complexity, DRY
- `REVIEW_AGENT.md` — orchestrates both into a full review
