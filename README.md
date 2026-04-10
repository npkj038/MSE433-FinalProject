# Flight Delay Buffer Optimization

## Overview

The pipeline:
1. Predicts whether a flight will be delayed (classification) and by how much (regression) using flight schedule and weather features
2. Solves a Linear Programme to allocate per-flight buffer times that minimize total cost
3. Extends the LP with aircraft connection constraints — flights sharing a turnaround window compete for a shared buffer pool
4. Evaluates four strategies: baseline, fixed buffer, optimized without connections, optimized with connections

---

## Repository Structure

```
├── data/
│   ├── Cancelled_Diverted_2023.csv       # Raw flight records (104,488 rows)
│   ├── weather_meteo_by_airport.csv      # Daily weather per airport (132,860 rows)
│   └── airports_geolocation.csv          # Airport reference table (364 rows)
│
├── flight_optimization.py             # Main pipeline (run this)
├── analysis.py                           # Visualization and analysis charts
│
├── outputs/
│   ├── flight_optimization_results.csv  # Per-flight results
│   ├── strategy_comparison.csv          # 4-strategy cost comparison
│   ├── connection_analysis.csv             # Connection constraint detail
│   ├── A_delay_patterns.png
│   ├── B_model_performance.png
│   ├── C_optimization_results.png
│   ├── D_weather_vs_delay.png
│   └── E_connection_impact.png
│
└── README.md
```

---

## Requirements

```bash
pip install requirements.txt
```

Python 3.8+ recommended. No other dependencies.

---

## Usage

### Step 1 — Run the main pipeline

```bash
python flight_optimization_v3.py
```

This runs the full pipeline end-to-end and saves three output CSVs. Expected runtime: 2–4 minutes on a standard laptop.

Update the paths at the top of the file if your data lives elsewhere:

```python
DATA_PATH    = "data/Cancelled_Diverted_2023.csv"
WEATHER_PATH = "data/weather_meteo_by_airport.csv"
```

### Step 2 — Run the analysis

```bash
python analysis.py
```

Generates five chart files covering delay patterns, model performance, optimization results, weather relationships, and connection constraint impact.

---

## Data Sources

| Dataset | Source |
|---------|--------|
| US 2023 civil flights, delays, meteo and aircrafts | Kaggle

Bordanova. (n.d.). US 2023 civil flights, delays, meteo and aircrafts. Kaggle. https://www.kaggle.com/datasets/bordanova/2023-us-civil-flights-delay-meteo-and-aircraft

---

## Versions

| File | Description |
|------|-------------|
| `flight_optimization.py` | v1 — baseline LP, no weather, no connections |
| `flight_optimization_v2.py` | v2 — adds connection constraints |
| `flight_optimization_v3.py` | v3 — adds weather features (use this one) |