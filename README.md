# NFL Dynasty Analytics

Dynasty fantasy football analytics platform applying institutional investment analysis to player valuation. Built around VORP (Value Over Replacement Player) rankings with DCF-style dynasty age adjustments.

**League:** 12-team Sleeper dynasty (ID: 1311316930342195200)

## Setup Checklist

Work through these in order. Each step builds on the last.

### Phase 1 — Environment & Version Control

- [ ] **Install Python 3.11+** — download from [python.org](https://www.python.org/downloads/). **Tick "Add Python to PATH"** during install.
- [ ] **Install Git** — download from [git-scm.com](https://git-scm.com/downloads)
- [ ] **Create a GitHub repo** — github.com → New Repository → `nfl-dynasty-analytics`, set to Private
- [ ] **Clone & open in PyCharm**
  ```bash
  git clone <your-repo-url>
  cd nfl-dynasty-analytics
  ```
- [ ] **Create virtual environment** — in PyCharm: `File → Settings → Project → Python Interpreter → Add Interpreter → Virtualenv`. Or via terminal:
  ```bash
  python -m venv .venv
  .venv\Scripts\activate     # Windows
  source .venv/bin/activate   # macOS/Linux
  ```
- [ ] **Install project dependencies**
  ```bash
  python -m pip install -r requirements.txt
  ```

### Phase 2 — Linting, Formatting & Code Quality

- [ ] **Install dev dependencies**
  ```bash
  python -m pip install -r requirements-dev.txt
  ```
- [ ] **Install pre-commit hooks**
  ```bash
  pre-commit install
  ```
- [ ] **Test the hooks work**
  ```bash
  pre-commit run --all-files
  ```

### Phase 3 — Configuration

- [ ] **Copy the environment template**
  ```bash
  cp .env.example .env
  ```
- [ ] **Edit `.env`** with your Sleeper username
- [ ] **(Optional)** Configure Infisical for production secrets management

### Phase 4 — Run the Dashboard

```bash
streamlit run src/dashboard/app.py
```

The dashboard opens at `http://localhost:8501`

## Project Structure

```
nfl-dynasty-analytics/
├── src/
│   ├── data/
│   │   ├── sleeper_client.py      # Sleeper API wrapper
│   │   └── nfl_stats.py           # nfl_data_py interface
│   ├── models/
│   │   ├── vorp.py                # VORP & Dynasty VORP calculations
│   │   ├── trade_analyzer.py      # Trade sentiment analysis
│   │   └── dropoff_predictor.py   # ML model for production decline
│   ├── pipeline/
│   │   └── (ETL & scheduling — coming soon)
│   └── dashboard/
│       ├── app.py                 # Streamlit draft helper
│       └── pages/
├── tests/
├── config/
│   └── settings.py                # Infisical / env var configuration
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Key Concepts

**VORP** — measures each player's value relative to the best freely available alternative at their position. The opportunity cost of NOT having that player.

**Dynasty VORP** — extends VORP with age-based decline projections. NPV applied to fantasy football. A 22-year-old WR with 15 VORP is worth far more than a 30-year-old RB with 15 VORP.

**Trade Sentiment** — analyses actual transaction behaviour from your league (and league mates' other leagues) to find information edges.

**Positional Scarcity** — concentration metrics showing which positions have the steepest talent drop-offs.

## Data Sources

| Source | Access | Data |
|--------|--------|------|
| Sleeper API | Free, no auth | League data, rosters, trades, drafts |
| nfl_data_py | Free, open source | Historical stats back to 1999 |
| Sleeper Trending | Free, no auth | Platform-wide add/drop trends |

## Roadmap

- [x] Core VORP calculation engine
- [x] Dynasty VORP with aging curves
- [x] Positional scarcity analysis
- [x] Streamlit draft helper dashboard
- [x] Sleeper API client with history traversal
- [ ] Live draft board with pick tracking
- [ ] Trade sentiment model
- [ ] Drop-off prediction pipeline
- [ ] NFL Fantasy app data integration
