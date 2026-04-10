"""
Flight Delay Optimization — Analysis & Visualizations
======================================================
Reads the outputs from flight_optimization_v3.py and produces
charts across 5 areas:

  A. Delay patterns     — by airline, airport, time of day, distance
  B. Model performance  — predicted vs actual, error distribution
  C. Optimization       — buffer vs cost vs delay, strategy comparison
  D. Weather vs delay   — scatter and binned relationships
  E. Connection impact  — binding constraints, buffer redistribution
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
RESULTS_PATH     = "./flight_optimization_results_v3.csv"
STRATEGY_PATH    = "./strategy_comparison_v3.csv"
CONN_PATH        = "./connection_analysis.csv"
FLIGHTS_RAW_PATH = "./Cancelled_Diverted_2023.csv"
WEATHER_PATH     = "./weather_meteo_by_airport.csv"
OUT_DIR          = "./"

# ── Style ──────────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
TEAL   = "#0D9488"
AMBER  = "#D97706"
CORAL  = "#E05C3A"
GRAY   = "#6B7280"
LIGHT  = "#F3F4F6"

plt.rcParams.update({
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "white",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.color"        : "#E5E7EB",
    "grid.linewidth"    : 0.6,
    "font.size"         : 11,
    "axes.titlesize"    : 13,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 11,
})

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(RESULTS_PATH)
df['FlightDate'] = pd.to_datetime(df['FlightDate'])
df['Day_Of_Week'] = df['FlightDate'].dt.dayofweek + 1
strat = pd.read_csv(STRATEGY_PATH)
conn = pd.read_csv(CONN_PATH)

# Re-merge weather for weather analysis
df_raw = pd.read_csv(FLIGHTS_RAW_PATH)
df_raw = df_raw[df_raw["Cancelled"] == 0.0].copy().reset_index(drop=True)
df_w = pd.read_csv(WEATHER_PATH)
df_with_weather = df_raw.merge(
    df_w.rename(columns={"time": "FlightDate", "airport_id": "Dep_Airport"}),
    on=["FlightDate", "Dep_Airport"], how="left"
)
df_with_weather["delay_min"] = df_with_weather["Dep_Delay"].clip(lower=0, upper=240)
for col in ["tavg","prcp","snow","wspd","pres"]:
    df_with_weather[col] = df_with_weather[col].fillna(df_with_weather[col].median())

print(f"  Results rows   : {len(df):,}")
print(f"  Connection pairs: {len(conn):,}")
print()

# ══════════════════════════════════════════════════════════════════════════
# A.  DELAY PATTERNS
# ══════════════════════════════════════════════════════════════════════════
print("A — Delay patterns...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("A  |  Delay Patterns", fontsize=15, fontweight="bold", x=0.02, ha="left")

# A1: Average delay by airline (top 12)
ax = axes[0, 0]
airline_delay = (
    df.groupby("Airline")["Dep_Delay"]
    .agg(["mean", "count"])
    .query("count >= 50")
    .sort_values("mean", ascending=True)
    .tail(12)
)
bars = ax.barh(airline_delay.index, airline_delay["mean"], color=BLUE, height=0.6)
ax.set_xlabel("Avg departure delay (min)")
ax.set_title("Avg delay by airline  (≥50 flights)")
ax.axvline(df["Dep_Delay"].mean(), color=CORAL, linestyle="--", linewidth=1.2, label=f"Overall avg ({df['Dep_Delay'].mean():.1f} min)")
ax.legend(fontsize=9)
for bar, val in zip(bars, airline_delay["mean"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}", va="center", fontsize=9)

# A2: Delay by time of day
ax = axes[0, 1]
time_order = ["Morning", "Afternoon", "Evening", "Night"]
time_data = df.groupby("DepTime_label")["Dep_Delay"].mean().reindex(time_order)
colors = [TEAL if v == time_data.max() else BLUE for v in time_data]
bars = ax.bar(time_data.index, time_data.values, color=colors, width=0.5)
ax.set_ylabel("Avg departure delay (min)")
ax.set_title("Avg delay by time of day")
for bar, val in zip(bars, time_data.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}", ha="center", fontsize=10)

# A3: Delay by day of week
ax = axes[1, 0]
day_labels = {1:"Mon", 2:"Tue", 3:"Wed", 4:"Thu", 5:"Fri", 6:"Sat", 7:"Sun"}
day_data = df.groupby("Day_Of_Week")["Dep_Delay"].mean()
day_data.index = [day_labels.get(d, d) for d in day_data.index]
colors = [CORAL if v == day_data.max() else BLUE for v in day_data.values]
bars = ax.bar(day_data.index, day_data.values, color=colors, width=0.5)
ax.set_ylabel("Avg departure delay (min)")
ax.set_title("Avg delay by day of week")
for bar, val in zip(bars, day_data.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}", ha="center", fontsize=10)

# A4: Delay distribution by distance type
ax = axes[1, 1]
dist_types = df["Distance_type"].unique()
colors_dist = [BLUE, TEAL, AMBER]
for i, (dtype, col) in enumerate(zip(["Short Haul", "Medium Haul", "Long Haul"], colors_dist)):
    subset = df[df["Distance_type"] == dtype]["Dep_Delay"].clip(-30, 150)
    ax.hist(subset, bins=40, alpha=0.55, color=col, label=dtype, density=True)
ax.set_xlabel("Departure delay (min)")
ax.set_ylabel("Density")
ax.set_title("Delay distribution by distance type")
ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT_DIR + "A_delay_patterns.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved A_delay_patterns.png")


# ══════════════════════════════════════════════════════════════════════════
# B.  MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
print("B — Model performance...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("B  |  Model Performance", fontsize=15, fontweight="bold", x=0.02, ha="left")

# B1: Predicted vs actual scatter
ax = axes[0]
sample = df.sample(min(2000, len(df)), random_state=42)
ax.scatter(sample["Predicted_Delay"], sample["Dep_Delay"],
           alpha=0.25, s=12, color=BLUE, rasterized=True)
lim = max(sample["Predicted_Delay"].max(), sample["Dep_Delay"].clip(upper=150).max()) + 5
ax.plot([0, lim], [0, lim], color=CORAL, linewidth=1.5, linestyle="--", label="Perfect prediction")
ax.set_xlabel("Predicted delay (min)")
ax.set_ylabel("Actual delay (min)")
ax.set_title("Predicted vs actual delay")
ax.set_xlim(0, lim); ax.set_ylim(-10, 150)
ax.legend(fontsize=9)

# B2: Prediction error distribution
ax = axes[1]
error = df["Dep_Delay"] - df["Predicted_Delay"]
ax.hist(error.clip(-100, 200), bins=60, color=TEAL, alpha=0.8, density=True)
ax.axvline(0, color=CORAL, linestyle="--", linewidth=1.5, label="Zero error")
ax.axvline(error.mean(), color=AMBER, linestyle="-", linewidth=1.5, label=f"Mean error ({error.mean():.1f} min)")
ax.set_xlabel("Actual − predicted (min)")
ax.set_ylabel("Density")
ax.set_title("Prediction error distribution")
ax.legend(fontsize=9)

# B3: Delay probability calibration
ax = axes[2]
df["prob_bin"] = pd.cut(df["Delay_Probability"], bins=10)
calib = df.groupby("prob_bin", observed=True).apply(
    lambda x: pd.Series({
        "mean_prob": x["Delay_Probability"].mean(),
        "actual_rate": (x["Dep_Delay"] > 15).mean(),
        "count": len(x)
    })
).reset_index()
ax.scatter(calib["mean_prob"], calib["actual_rate"],
           s=calib["count"] / calib["count"].max() * 300,
           color=BLUE, alpha=0.8, zorder=3)
ax.plot([0, 1], [0, 1], color=CORAL, linestyle="--", linewidth=1.5, label="Perfect calibration")
ax.set_xlabel("Predicted probability of delay")
ax.set_ylabel("Actual delay rate")
ax.set_title("Classifier calibration\n(dot size = sample count)")
ax.legend(fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT_DIR + "B_model_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved B_model_performance.png")


# ══════════════════════════════════════════════════════════════════════════
# C.  OPTIMIZATION RESULTS
# ══════════════════════════════════════════════════════════════════════════
print("C — Optimization results...")
fig = plt.figure(figsize=(15, 10))
fig.suptitle("C  |  Optimization Results", fontsize=15, fontweight="bold", x=0.02, ha="left")
gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# C1: Strategy comparison bar chart (total cost)
ax = fig.add_subplot(gs[0, 0])
strategies = strat["Strategy"].str.replace(" [(].*[)]", "", regex=True)
costs = strat["Total Cost"]
bar_colors = [GRAY, BLUE, TEAL, CORAL]
bars = ax.bar(range(len(strategies)), costs / 1e6, color=bar_colors, width=0.6)
ax.set_xticks(range(len(strategies)))
ax.set_xticklabels(["Baseline", "Fixed\n15 min", "Optimized\nno conn.", "Optimized\nwith conn."], fontsize=9)
ax.set_ylabel("Total cost (millions)")
ax.set_title("Total cost by strategy")
for bar, val in zip(bars, costs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val/1e6:.2f}M", ha="center", fontsize=9)

# C2: Buffer vs residual delay tradeoff
ax = fig.add_subplot(gs[0, 1])
total_buffers  = strat["Total Buffer (min)"]
total_residual = strat["Total Residual Delay"]
sc = ax.scatter(total_buffers / 1e3, total_residual / 1e3,
                s=200, c=bar_colors, zorder=3)
for i, row in strat.iterrows():
    label = ["Baseline", "Fixed", "Opt (no conn)", "Opt (conn)"][i]
    ax.annotate(label, (row["Total Buffer (min)"]/1e3, row["Total Residual Delay"]/1e3),
                textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.set_xlabel("Total buffer added (000s min)")
ax.set_ylabel("Total residual delay (000s min)")
ax.set_title("Buffer vs residual delay tradeoff")

# C3: % flights still delayed
ax = fig.add_subplot(gs[0, 2])
pct_delayed = strat["% Still Delayed"].str.rstrip("%").astype(float)
bars = ax.bar(range(len(strategies)), pct_delayed, color=bar_colors, width=0.6)
ax.set_xticks(range(len(strategies)))
ax.set_xticklabels(["Baseline", "Fixed\n15 min", "Optimized\nno conn.", "Optimized\nwith conn."], fontsize=9)
ax.set_ylabel("% flights still delayed")
ax.set_title("Flights still delayed by strategy")
for bar, val in zip(bars, pct_delayed):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", fontsize=9)

# C4: Per-flight buffer distribution (optimized with connections)
ax = fig.add_subplot(gs[1, 0])
ax.hist(df["Buffer_With_Connections"], bins=35, color=CORAL, alpha=0.8)
ax.axvline(df["Buffer_With_Connections"].mean(), color=BLUE, linewidth=1.5,
           linestyle="--", label=f"Mean ({df['Buffer_With_Connections'].mean():.1f} min)")
ax.set_xlabel("Optimal buffer (min)")
ax.set_ylabel("Number of flights")
ax.set_title("Buffer distribution\n(optimized with connections)")
ax.legend(fontsize=9)

# C5: Cost saving per flight
ax = fig.add_subplot(gs[1, 1])
ax.hist(df["Cost_Saving"].clip(-60, 100), bins=50, color=TEAL, alpha=0.8)
ax.axvline(0, color=CORAL, linestyle="--", linewidth=1.5)
ax.axvline(df["Cost_Saving"].mean(), color=AMBER, linestyle="-", linewidth=1.5,
           label=f"Mean saving ({df['Cost_Saving'].mean():.1f})")
ax.set_xlabel("Cost saving per flight (units)")
ax.set_ylabel("Number of flights")
ax.set_title("Per-flight cost saving\n(optimized vs baseline)")
ax.legend(fontsize=9)

# C6: Top airports by total cost saving
ax = fig.add_subplot(gs[1, 2])
airport_saving = df.groupby("Dep_Airport")["Cost_Saving"].sum().sort_values().tail(12)
ax.barh(airport_saving.index, airport_saving.values, color=BLUE, height=0.6)
ax.set_xlabel("Total cost saving (units)")
ax.set_title("Top 12 airports\nby total cost saving")

plt.savefig(OUT_DIR + "C_optimization_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved C_optimization_results.png")


# ══════════════════════════════════════════════════════════════════════════
# D.  WEATHER VS DELAY
# ══════════════════════════════════════════════════════════════════════════
print("D — Weather vs delay...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("D  |  Weather vs Delay", fontsize=15, fontweight="bold", x=0.02, ha="left")

wdf = df_with_weather.copy()

weather_vars = [
    ("tavg",  "Avg temperature (°C)",       BLUE),
    ("prcp",  "Precipitation (mm)",          TEAL),
    ("wspd",  "Wind speed (km/h)",           AMBER),
    ("pres",  "Atmospheric pressure (hPa)",  CORAL),
    ("snow",  "Snowfall (mm)",               GRAY),
]

for idx, (col, label, color) in enumerate(weather_vars):
    ax = axes[idx // 3][idx % 3]
    # Bin the weather variable into 10 equal-frequency bins
    wdf["bin"] = pd.qcut(wdf[col], q=10, duplicates="drop")
    binned = wdf.groupby("bin", observed=True)["delay_min"].agg(["mean", "count", "sem"]).reset_index()
    x_vals = range(len(binned))
    ax.bar(x_vals, binned["mean"], color=color, alpha=0.75, width=0.7)
    ax.errorbar(x_vals, binned["mean"], yerr=binned["sem"]*1.96,
                fmt="none", color="black", capsize=3, linewidth=1)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([f"{b.mid:.0f}" for b in binned["bin"]], rotation=45, fontsize=8)
    ax.set_xlabel(label)
    ax.set_ylabel("Avg delay (min)")
    ax.set_title(f"Delay vs {col}")

# D6: Heatmap of delay by temp x precipitation bucket
ax = axes[1, 2]
wdf["temp_bin"] = pd.cut(wdf["tavg"], bins=[-30, 0, 10, 20, 30, 50], labels=["<0", "0-10", "10-20", "20-30", ">30"])
wdf["prcp_bin"] = pd.cut(wdf["prcp"], bins=[-1, 0, 2, 10, 1000], labels=["None", "Light", "Moderate", "Heavy"])
heat = wdf.pivot_table(values="delay_min", index="temp_bin", columns="prcp_bin", aggfunc="mean")
im = ax.imshow(heat.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(heat.columns))); ax.set_xticklabels(heat.columns, fontsize=9)
ax.set_yticks(range(len(heat.index)));   ax.set_yticklabels(heat.index, fontsize=9)
ax.set_xlabel("Precipitation level")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Avg delay: temperature × precipitation")
plt.colorbar(im, ax=ax, label="Avg delay (min)", shrink=0.8)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        val = heat.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if val > 40 else "black")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT_DIR + "D_weather_vs_delay.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved D_weather_vs_delay.png")


# ══════════════════════════════════════════════════════════════════════════
# E.  CONNECTION CONSTRAINT IMPACT
# ══════════════════════════════════════════════════════════════════════════
print("E — Connection constraint impact...")
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("E  |  Connection Constraint Impact", fontsize=15, fontweight="bold", x=0.02, ha="left")

# E1: Buffer change due to connections (upstream flight i)
ax = axes[0, 0]
buf_change_i = conn["buffer_i_conn"] - conn["buffer_i_no_conn"]
ax.hist(buf_change_i.clip(-35, 5), bins=40, color=BLUE, alpha=0.8)
ax.axvline(0, color=CORAL, linestyle="--", linewidth=1.5)
ax.axvline(buf_change_i.mean(), color=AMBER, linestyle="-", linewidth=1.5,
           label=f"Mean change ({buf_change_i.mean():.1f} min)")
ax.set_xlabel("Buffer change: upstream flight (min)")
ax.set_ylabel("Number of connections")
ax.set_title("Buffer change — upstream flight (i)")
ax.legend(fontsize=9)

# E2: Buffer change for downstream flight j
ax = axes[0, 1]
buf_change_j = conn["buffer_j_conn"] - conn["buffer_j_no_conn"]
ax.hist(buf_change_j.clip(-35, 5), bins=40, color=TEAL, alpha=0.8)
ax.axvline(0, color=CORAL, linestyle="--", linewidth=1.5)
ax.axvline(buf_change_j.mean(), color=AMBER, linestyle="-", linewidth=1.5,
           label=f"Mean change ({buf_change_j.mean():.1f} min)")
ax.set_xlabel("Buffer change: downstream flight (min)")
ax.set_ylabel("Number of connections")
ax.set_title("Buffer change — downstream flight (j)")
ax.legend(fontsize=9)

# E3: Combined buffer per pair - with vs without constraint
ax = axes[1, 0]
combined_no_conn = conn["buffer_i_no_conn"] + conn["buffer_j_no_conn"]
combined_conn    = conn["buffer_i_conn"]    + conn["buffer_j_conn"]
ax.scatter(combined_no_conn, combined_conn, alpha=0.15, s=8, color=BLUE, rasterized=True)
lim = max(combined_no_conn.max(), combined_conn.max()) + 2
ax.plot([0, lim], [0, lim], color=GRAY, linestyle="--", linewidth=1.2, label="No change")
ax.axhline(45, color=CORAL, linestyle="-", linewidth=1.2, label="Pool cap (45 min)")
ax.set_xlabel("Combined buffer: no connections (min)")
ax.set_ylabel("Combined buffer: with connections (min)")
ax.set_title("Pair-level buffer: before vs after constraint")
ax.legend(fontsize=9)

# E4: Top airports by number of binding connections
ax = axes[1, 1]
# A connection is effectively constrained if the combined buffer dropped
conn["constrained"] = combined_conn < (combined_no_conn - 0.5)
top_airports = (
    conn[conn["constrained"]]
    .groupby("dep_airport_j")
    .size()
    .sort_values(ascending=True)
    .tail(12)
)
ax.barh(top_airports.index, top_airports.values, color=CORAL, height=0.6)
ax.set_xlabel("Constrained connection pairs")
ax.set_title("Top airports with constrained\nturnaround pairs")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT_DIR + "E_connection_impact.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved E_connection_impact.png")

print()
print("=" * 50)
print("All charts saved to ./")
print("  A_delay_patterns.png")
print("  B_model_performance.png")
print("  C_optimization_results.png")
print("  D_weather_vs_delay.png")
print("  E_connection_impact.png")
print("=" * 50)