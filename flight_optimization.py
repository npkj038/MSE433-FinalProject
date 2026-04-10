"""
Flight Delay Buffer Optimization Pipeline — v3 (with Weather Features)
=============================================================================
Predictive + Prescriptive model for optimal departure buffer allocation.

NEW in v3:
  - Weather data merged from weather_meteo_by_airport.csv on Dep_Airport + date
  - Weather features (tavg, prcp, snow, wspd, pres, wdir) added to both ML models
  - Aircraft connection constraints retained from v2

Steps:
  1. Data loading & feature engineering
  2. ML Model 1 – classify delay > 15 min (Random Forest)
  3. ML Model 2 – regress expected delay (Random Forest)
  4a. Optimization (no connections) – baseline LP
  4b. Optimization (with connections) – extended LP
  5. Evaluation – compare 4 strategies
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from scipy.optimize import linprog
from scipy.sparse import eye as speye, hstack as sphstack, vstack as spvstack, csc_matrix
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
DATA_PATH    = "./data/Cancelled_Diverted_2023.csv"
WEATHER_PATH = "./data/weather_meteo_by_airport.csv"
C_BUFFER    = 1.0    # cost per minute of buffer added
C_DELAY     = 3.0    # cost per minute of residual delay
MAX_BUFFER  = 30     # hard cap on buffer per flight (minutes)
MIN_TURNAROUND  = 30   # minimum required turnaround time between connected flights (minutes)
SLOT_MINUTES    = 180  # approximate minutes per time-of-day slot
TURNAROUND_POOL = 45   # shared buffer pool per connection pair: x_i + x_j <= TURNAROUND_POOL

# ─────────────────────────────────────────────
# 1. LOAD & FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — DATA LOADING, WEATHER MERGE & FEATURE ENGINEERING")
print("=" * 65)

df_raw = pd.read_csv(DATA_PATH)
print(f"Raw records        : {len(df_raw):,}")

df = df_raw[df_raw["Cancelled"] == 0.0].copy().reset_index(drop=True)
print(f"Non-cancelled used : {len(df):,}")

# ── Weather merge ──────────────────────────────────────────────────────────
df_w = pd.read_csv(WEATHER_PATH)
df = df.merge(
    df_w.rename(columns={"time": "FlightDate", "airport_id": "Dep_Airport"}),
    on=["FlightDate", "Dep_Airport"],
    how="left"
)
WEATHER_COLS = ["tavg", "prcp", "snow", "wspd", "pres", "wdir"]
for col in WEATHER_COLS:
    df[col] = df[col].fillna(df[col].median())
coverage = df["tavg"].notna().mean()
print(f"Weather merge coverage: {coverage:.1%}")
print(f"Weather features added: {WEATHER_COLS}")
# ──────────────────────────────────────────────────────────────────────────

df["Dep_Delay"] = df["Dep_Delay"].clip(upper=240)
df["delay_flag"] = (df["Dep_Delay"] > 15).astype(int)
df["delay_min"]  = df["Dep_Delay"].clip(lower=0)

print(f"Delayed >15 min    : {df['delay_flag'].mean():.1%}")
print(f"Mean delay         : {df['Dep_Delay'].mean():.1f} min")

le_a = LabelEncoder(); le_t = LabelEncoder(); le_d = LabelEncoder()
df["Airline_enc"]      = le_a.fit_transform(df["Airline"])
df["DepTime_enc"]      = le_t.fit_transform(df["DepTime_label"])
df["DistanceType_enc"] = le_d.fit_transform(df["Distance_type"])

FEATURES = [
    "Day_Of_Week", "Airline_enc", "DepTime_enc", "DistanceType_enc",
    "tavg", "prcp", "snow", "wspd", "pres", "wdir"
]

X      = df[FEATURES]
y_clf  = df["delay_flag"]
y_reg  = df["delay_min"]

X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
    X, y_clf, y_reg, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")


# ─────────────────────────────────────────────
# 2. ML MODEL 1 — CLASSIFICATION
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — ML MODEL 1: DELAY CLASSIFICATION (>15 min)")
print("=" * 65)

clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
clf.fit(X_train, yc_train)

yc_pred  = clf.predict(X_test)
yc_proba = clf.predict_proba(X_test)[:, 1]

print(classification_report(yc_test, yc_pred, target_names=["On-time", "Delayed"]))
print(f"ROC-AUC: {roc_auc_score(yc_test, yc_proba):.4f}")

feat_imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature Importances:")
for feat, imp in feat_imp.items():
    bar = "█" * int(imp * 40)
    print(f"  {feat:<28} {imp:.4f}  {bar}")


# ─────────────────────────────────────────────
# 3. ML MODEL 2 — REGRESSION
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 — ML MODEL 2: DELAY REGRESSION (minutes)")
print("=" * 65)

reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
reg.fit(X_train, yr_train)

yr_pred = reg.predict(X_test)
print(f"MAE  : {mean_absolute_error(yr_test, yr_pred):.2f} min")
print(f"RMSE : {np.sqrt(mean_squared_error(yr_test, yr_pred)):.2f} min")
print(f"R²   : {r2_score(yr_test, yr_pred):.4f}")

# Predicted delays for ALL flights (used by optimizer)
d_pred = reg.predict(X)
n = len(d_pred)


# ─────────────────────────────────────────────
# 4. BUILD CONNECTION PAIRS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4 — BUILDING AIRCRAFT CONNECTION GRAPH")
print("=" * 65)

time_order = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
df["time_rank"] = df["DepTime_label"].map(time_order)
df["orig_idx"]  = df.index

# Match: flight i arrives at airport A, flight j departs from A later same day
df_arr = df[["orig_idx", "FlightDate", "Arr_Airport", "time_rank"]].copy()
df_dep = df[["orig_idx", "FlightDate", "Dep_Airport", "time_rank"]].copy()
df_arr.columns = ["arr_idx", "FlightDate", "airport", "arr_rank"]
df_dep.columns = ["dep_idx", "FlightDate", "airport", "dep_rank"]

merged = pd.merge(df_arr, df_dep, on=["FlightDate", "airport"])
# Only forward connections (next slot, not same or earlier)
connected = merged[merged["dep_rank"] > merged["arr_rank"]].copy()
connected["gap_slots"] = connected["dep_rank"] - connected["arr_rank"]
# Each arriving flight connects to the nearest subsequent departure
connected = (
    connected.sort_values("gap_slots")
    .groupby("arr_idx")
    .first()
    .reset_index()
)

pairs = list(zip(connected["arr_idx"].astype(int), connected["dep_idx"].astype(int),
                 connected["gap_slots"].astype(int)))

print(f"Connection pairs found : {len(pairs):,}")
print(f"Flights with outgoing connection: {connected['arr_idx'].nunique():,} ({connected['arr_idx'].nunique()/n:.1%})")
print(f"Flights receiving a connection  : {connected['dep_idx'].nunique():,} ({connected['dep_idx'].nunique()/n:.1%})")


# ─────────────────────────────────────────────
# 5a. LP WITHOUT CONNECTIONS (reference)
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5a — OPTIMIZATION: NO CONNECTIONS (reference)")
print("=" * 65)

def solve_lp(d_pred, n, pairs=None, pool=TURNAROUND_POOL,
             max_buf=MAX_BUFFER, c_b=C_BUFFER, c_d=C_DELAY):
    """
    Variables: [x_0..x_{n-1}, z_0..z_{n-1}]
      x_i = buffer allocated to flight i   (0 <= x_i <= max_buf)
      z_i = residual delay after buffer    (z_i >= 0)

    Objective: min  sum c_b*x_i + c_d*z_i

    Core constraints (every flight i):
      z_i >= d_i - x_i   =>   -x_i - z_i <= -d_i

    Connection constraints (every pair i->j sharing a turnaround):
      The two connected flights share a fixed turnaround window.
      Their combined buffers cannot exceed that pool:
        x_i + x_j <= TURNAROUND_POOL
      This forces the optimizer to CHOOSE which flight to protect.
    """
    c_obj = np.concatenate([c_b * np.ones(n), c_d * np.ones(n)])

    I = speye(n, format='csc')

    # Core: -x_i - z_i <= -d_i
    A_core = sphstack([-I, -I], format='csc')
    b_core = -np.clip(d_pred, 0, None)

    if pairs:
        m_conn  = len(pairs)
        col_xi  = np.array([p[0] for p in pairs])
        col_xj  = np.array([p[1] for p in pairs])
        row_ids = np.arange(m_conn)

        # x_i + x_j <= pool  =>  [+1 on x_i, +1 on x_j, 0 on z-block]
        A_conn_x = csc_matrix(
            (np.ones(2 * m_conn),
             (np.concatenate([row_ids, row_ids]),
              np.concatenate([col_xi, col_xj]))),
            shape=(m_conn, n)
        )
        A_conn_z = csc_matrix((m_conn, n))
        A_conn   = sphstack([A_conn_x, A_conn_z], format='csc')
        b_conn   = np.full(m_conn, float(pool))

        A_ub = spvstack([A_core, A_conn], format='csc')
        b_ub = np.concatenate([b_core, b_conn])
    else:
        A_ub = A_core
        b_ub = b_core

    x_lb = np.zeros(2 * n)
    x_ub = np.concatenate([np.full(n, float(max_buf)), np.full(n, np.inf)])
    bounds = list(zip(x_lb, x_ub))

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    return result


res_no_conn = solve_lp(d_pred, n, pairs=None)
x_no_conn = res_no_conn.x[:n]
z_no_conn = res_no_conn.x[n:]
print(f"Status     : {res_no_conn.message}")
print(f"Mean buffer: {x_no_conn.mean():.2f} min")
print(f"Max buffer : {x_no_conn.max():.2f} min")


# ─────────────────────────────────────────────
# 5b. LP WITH CONNECTION CONSTRAINTS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5b — OPTIMIZATION: WITH CONNECTION CONSTRAINTS")
print("=" * 65)
print(f"Min turnaround enforced: {MIN_TURNAROUND} min")
print(f"Shared turnaround pool : {TURNAROUND_POOL} min per connection pair")

res_conn = solve_lp(d_pred, n, pairs=pairs, pool=TURNAROUND_POOL)
x_conn = res_conn.x[:n]
z_conn = res_conn.x[n:]
print(f"Status     : {res_conn.message}")
print(f"Mean buffer: {x_conn.mean():.2f} min")
print(f"Max buffer : {x_conn.max():.2f} min")

# How many connection constraints were binding?
binding = 0
for i, j, gap_slots in pairs:
    if abs((x_conn[i] + x_conn[j]) - TURNAROUND_POOL) < 0.5:
        binding += 1
print(f"Binding connection constraints: {binding} / {len(pairs)} ({binding/len(pairs):.1%})")


# ─────────────────────────────────────────────
# 6. EVALUATION — 4-STRATEGY COMPARISON
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 — STRATEGY COMPARISON")
print("=" * 65)

actual = df["delay_min"].values

def evaluate(buffers, actual, label):
    residual   = np.maximum(0, actual - buffers)
    tot_buf    = buffers.sum()
    tot_del    = residual.sum()
    tot_cost   = C_BUFFER * tot_buf + C_DELAY * tot_del
    n_delayed  = (residual > 0).sum()
    return {
        "Strategy"              : label,
        "Avg Buffer (min)"      : round(buffers.mean(), 2),
        "Total Buffer (min)"    : round(tot_buf, 0),
        "Avg Residual Delay"    : round(residual.mean(), 2),
        "Total Residual Delay"  : round(tot_del, 0),
        "Flights Still Delayed" : int(n_delayed),
        "% Still Delayed"       : f"{n_delayed/len(actual):.1%}",
        "Total Cost"            : round(tot_cost, 0),
    }

s1 = evaluate(np.zeros(n),       actual, "Baseline (no buffer)")
s2 = evaluate(np.full(n, 15.0),  actual, "Fixed 15-min buffer")
s3 = evaluate(x_no_conn,         actual, "Optimized (no connections)")
s4 = evaluate(x_conn,            actual, "Optimized (with connections)")

results_df = pd.DataFrame([s1, s2, s3, s4]).set_index("Strategy")
print("\n" + results_df.T.to_string())

# Cost delta table
base_cost = s1["Total Cost"]
print("\n💰 Cost savings vs Baseline:")
for s in [s2, s3, s4]:
    delta = s["Total Cost"] - base_cost
    pct   = delta / base_cost * 100
    print(f"   {s['Strategy']:<35} {pct:+.1f}%  ({delta:+,.0f} units)")

print("\n💡 Impact of adding connection constraints:")
delta_conn = s4["Total Cost"] - s3["Total Cost"]
pct_conn   = delta_conn / s3["Total Cost"] * 100
print(f"   Cost change: {pct_conn:+.2f}%  ({delta_conn:+,.0f} units)")
print(f"   Avg buffer change: {x_conn.mean()-x_no_conn.mean():+.2f} min")
print(f"   Flights still delayed: {s4['Flights Still Delayed']} vs {s3['Flights Still Delayed']}")


# ─────────────────────────────────────────────
# 7. CONNECTION CONSTRAINT ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7 — CONNECTION CONSTRAINT DEEP DIVE")
print("=" * 65)

conn_analysis = []
for i, j, gap_slots in pairs:
    slack = gap_slots * SLOT_MINUTES - MIN_TURNAROUND
    conn_analysis.append({
        "flight_i"          : i,
        "flight_j"          : j,
        "gap_slots"         : gap_slots,
        "slack_available"   : slack,
        "buffer_i_no_conn"  : round(x_no_conn[i], 2),
        "buffer_j_no_conn"  : round(x_no_conn[j], 2),
        "buffer_i_conn"     : round(x_conn[i], 2),
        "buffer_j_conn"     : round(x_conn[j], 2),
        "constraint_rhs"    : slack,
        "constraint_lhs"    : round(x_conn[j] - x_conn[i], 2),
        "binding"           : abs((x_conn[j] - x_conn[i]) - slack) < 0.5,
        "dep_airport_j"     : df.loc[j, "Dep_Airport"],
        "airline_j"         : df.loc[j, "Airline"],
    })

conn_df = pd.DataFrame(conn_analysis)
print(f"\nConnections where constraint is BINDING: {conn_df['binding'].sum()}")
print(f"\nTop airports by binding constraint count:")
binding_df = conn_df[conn_df["binding"]]
if len(binding_df) > 0:
    print(binding_df["dep_airport_j"].value_counts().head(10).to_string())

print(f"\nBuffer reduction from adding connections (connected flights only):")
conn_df["buf_j_change"] = conn_df["buffer_j_conn"] - conn_df["buffer_j_no_conn"]
print(f"  Mean buffer change on connected departures: {conn_df['buf_j_change'].mean():+.2f} min")
print(f"  Flights where buffer was REDUCED: {(conn_df['buf_j_change'] < -0.1).sum()}")
print(f"  Flights where buffer was INCREASED: {(conn_df['buf_j_change'] > 0.1).sum()}")


# ─────────────────────────────────────────────
# 8. SAVE OUTPUTS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 8 — SAVING OUTPUTS")
print("=" * 65)

output_df = df[["FlightDate", "Airline", "Dep_Airport", "Arr_Airport",
                "DepTime_label", "Distance_type", "Dep_Delay"]].copy()
output_df["Predicted_Delay"]       = np.round(d_pred, 2)
output_df["Delay_Probability"]     = np.round(clf.predict_proba(X)[:, 1], 4)
output_df["Buffer_No_Connections"] = np.round(x_no_conn, 2)
output_df["Buffer_With_Connections"] = np.round(x_conn, 2)
output_df["Residual_Delay_Baseline"] = np.round(np.maximum(0, actual), 2)
output_df["Residual_Delay_Optimized"] = np.round(np.maximum(0, actual - x_conn), 2)
output_df["Cost_Baseline"]          = np.round(C_DELAY * np.maximum(0, actual), 2)
output_df["Cost_Optimized"]         = np.round(C_BUFFER * x_conn + C_DELAY * np.maximum(0, actual - x_conn), 2)
output_df["Cost_Saving"]            = np.round(output_df["Cost_Baseline"] - output_df["Cost_Optimized"], 2)
output_df["Has_Connection"]         = output_df.index.isin(connected["arr_idx"].values)

output_df.to_csv("./flight_optimization_results.csv", index=False)
print("  ✅ Per-flight results → flight_optimization_results.csv")

results_df.to_csv("./strategy_comparison_v3")
print("  ✅ Strategy comparison → strategy_comparison.csv")

conn_df.to_csv("./connection_analysis.csv", index=False)
print("  ✅ Connection analysis → connection_analysis.csv")

print("\n" + "=" * 65)
print("PIPELINE v3 COMPLETE")
print("=" * 65)

# The use of GEN-AI was used to assist in coding the pipeline, especially the optimization and connection constraint logic. The code was then manually reviewed, tested, and refined to ensure correctness and clarity.