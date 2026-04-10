import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

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

# ── Load ───────────────────────────────────────────────────────────────────
df_raw = pd.read_csv('./data/Cancelled_Diverted_2023.csv')
df_w   = pd.read_csv('./data/weather_meteo_by_airport.csv')

df_nc = df_raw[df_raw['Cancelled'] == 0.0].copy().reset_index(drop=True)
df_nc['Dep_Delay'] = df_nc['Dep_Delay'].clip(upper=240)
df_nc['delay_flag'] = (df_nc['Dep_Delay'] > 15).astype(int)
df_nc['FlightDate_dt'] = pd.to_datetime(df_nc['FlightDate'])
df_nc['Month'] = df_nc['FlightDate_dt'].dt.month

merged = df_nc.merge(
    df_w.rename(columns={'time':'FlightDate','airport_id':'Dep_Airport'}),
    left_on=['FlightDate','Dep_Airport'], right_on=['FlightDate','Dep_Airport'], how='left'
)
for col in ['tavg','prcp','snow','wspd','pres']:
    merged[col] = merged[col].fillna(merged[col].median())

# ══════════════════════════════════════════════════════════════════════════
# CHART 1 — Dataset composition + delay distribution
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Initial Data Analysis  |  Dataset Overview", fontsize=14,
             fontweight="bold", x=0.02, ha="left")

# 1a: Pie — cancelled vs non-cancelled
ax = axes[0]
sizes  = [87936, 16552]
labels = ['Cancelled\n87,936 (84.2%)', 'Non-cancelled\n16,552 (15.8%)']
colors = [CORAL, BLUE]
wedges, texts = ax.pie(sizes, labels=labels, colors=colors,
                       startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
for t in texts:
    t.set_fontsize(10)
ax.set_title("Dataset composition\n(104,488 total flights)")

# 1b: Delay status breakdown (non-cancelled only)
ax = axes[1]
on_time  = (df_nc['Dep_Delay'] <= 15).sum()
delayed  = (df_nc['Dep_Delay'] >  15).sum()
bars = ax.bar(['On time / minor\ndelay (≤15 min)', 'Delayed\n(>15 min)'],
              [on_time, delayed], color=[TEAL, CORAL], width=0.5)
ax.set_ylabel("Number of flights")
ax.set_title("Delay status\n(non-cancelled flights only)")
for bar, val in zip(bars, [on_time, delayed]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
            f"{val:,}\n({val/len(df_nc)*100:.1f}%)", ha='center', fontsize=10)
ax.set_ylim(0, 13000)

# 1c: Departure delay distribution
ax = axes[2]
delays = df_nc['Dep_Delay'].clip(-30, 150)
ax.hist(delays, bins=60, color=BLUE, alpha=0.8, edgecolor='white', linewidth=0.3)
ax.axvline(df_nc['Dep_Delay'].mean(), color=CORAL, linewidth=1.8, linestyle='--',
           label=f"Mean: {df_nc['Dep_Delay'].mean():.1f} min")
ax.axvline(df_nc['Dep_Delay'].median(), color=AMBER, linewidth=1.8, linestyle='-',
           label=f"Median: {df_nc['Dep_Delay'].median():.0f} min")
ax.axvline(15, color=GRAY, linewidth=1.2, linestyle=':', label="15-min threshold")
ax.set_xlabel("Departure delay (min)")
ax.set_ylabel("Number of flights")
ax.set_title("Departure delay distribution\n(non-cancelled, clipped at 150 min)")
ax.legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('/mnt/user-data/outputs/IA1_dataset_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved IA1_dataset_overview.png")

# ══════════════════════════════════════════════════════════════════════════
# CHART 2 — Delay by airline, time, day, month
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Initial Data Analysis  |  Delay Patterns Across Key Dimensions",
             fontsize=14, fontweight="bold", x=0.02, ha="left")

# 2a: Airline avg delay (all airlines)
ax = axes[0, 0]
airline_stats = (df_nc.groupby('Airline')['Dep_Delay']
                 .agg(['mean','count'])
                 .query('count >= 50')
                 .sort_values('mean', ascending=True))
colors_a = [CORAL if v > df_nc['Dep_Delay'].mean() else BLUE for v in airline_stats['mean']]
bars = ax.barh(airline_stats.index, airline_stats['mean'], color=colors_a, height=0.65)
ax.axvline(df_nc['Dep_Delay'].mean(), color=GRAY, linewidth=1.2,
           linestyle='--', label=f"Overall avg ({df_nc['Dep_Delay'].mean():.1f} min)")
ax.set_xlabel("Avg departure delay (min)")
ax.set_title("Average delay by airline  (≥50 flights)")
ax.legend(fontsize=9)
for bar, val in zip(bars, airline_stats['mean']):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}", va='center', fontsize=9)

# 2b: Time of day
ax = axes[0, 1]
time_order = ['Morning','Afternoon','Evening','Night']
time_stats = df_nc.groupby('DepTime_label')['Dep_Delay'].agg(['mean','sem']).reindex(time_order)
bar_colors = [CORAL if v == time_stats['mean'].max() else
              TEAL  if v == time_stats['mean'].min() else BLUE
              for v in time_stats['mean']]
bars = ax.bar(time_stats.index, time_stats['mean'], color=bar_colors,
              width=0.5, yerr=time_stats['sem']*1.96, capsize=4,
              error_kw={'linewidth':1.2, 'color':'#374151'})
ax.set_ylabel("Avg departure delay (min)")
ax.set_title("Average delay by time of day")
for bar, val in zip(bars, time_stats['mean']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{val:.1f}", ha='center', fontsize=10)

# 2c: Day of week
ax = axes[1, 0]
day_labels = {1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat',7:'Sun'}
df_nc['DayLabel'] = df_nc['FlightDate_dt'].dt.dayofweek.map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
day_order = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
day_stats = df_nc.groupby('DayLabel')['Dep_Delay'].agg(['mean','sem']).reindex(day_order)
bar_colors_d = [CORAL if v == day_stats['mean'].max() else BLUE for v in day_stats['mean']]
bars = ax.bar(day_stats.index, day_stats['mean'], color=bar_colors_d,
              width=0.5, yerr=day_stats['sem']*1.96, capsize=4,
              error_kw={'linewidth':1.2, 'color':'#374151'})
ax.set_ylabel("Avg departure delay (min)")
ax.set_title("Average delay by day of week")
for bar, val in zip(bars, day_stats['mean']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f"{val:.1f}", ha='center', fontsize=10)

# 2d: Monthly trend
ax = axes[1, 1]
month_labels = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
month_stats = df_nc.groupby('Month')['Dep_Delay'].agg(['mean','sem','count'])
month_stats.index = [month_labels[m] for m in month_stats.index]
ax.plot(month_stats.index, month_stats['mean'], color=BLUE,
        linewidth=2, marker='o', markersize=6, zorder=3)
ax.fill_between(month_stats.index,
                month_stats['mean'] - month_stats['sem']*1.96,
                month_stats['mean'] + month_stats['sem']*1.96,
                alpha=0.15, color=BLUE)
ax.axhline(df_nc['Dep_Delay'].mean(), color=GRAY, linewidth=1.2,
           linestyle='--', label=f"Annual avg ({df_nc['Dep_Delay'].mean():.1f} min)")
ax2 = ax.twinx()
ax2.bar(month_stats.index, month_stats['count'], color=GRAY, alpha=0.2, zorder=1)
ax2.set_ylabel("Flight count", color=GRAY, fontsize=10)
ax2.tick_params(axis='y', labelcolor=GRAY)
ax.set_ylabel("Avg departure delay (min)")
ax.set_title("Average delay by month\n(line = avg delay, bars = flight count)")
ax.legend(fontsize=9)
ax.set_zorder(ax2.get_zorder() + 1)
ax.patch.set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/mnt/user-data/outputs/IA2_delay_patterns.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved IA2_delay_patterns.png")

# ══════════════════════════════════════════════════════════════════════════
# CHART 3 — Weather relationships
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Initial Data Analysis  |  Weather vs Departure Delay",
             fontsize=14, fontweight="bold", x=0.02, ha="left")

merged['delay_min'] = merged['Dep_Delay'].clip(lower=0)

# 3a: Temp vs delay (scatter + binned mean)
ax = axes[0, 0]
sample = merged.sample(3000, random_state=42)
ax.scatter(sample['tavg'], sample['Dep_Delay'].clip(-10,150),
           alpha=0.12, s=8, color=BLUE, rasterized=True)
merged['temp_bin'] = pd.cut(merged['tavg'], bins=10)
binned = merged.groupby('temp_bin', observed=True)['delay_min'].agg(['mean','sem']).reset_index()
xvals = [b.mid for b in binned['temp_bin']]
ax.plot(xvals, binned['mean'], color=CORAL, linewidth=2.2,
        marker='o', markersize=6, zorder=5, label='Binned mean')
ax.set_xlabel("Avg temperature (°C)")
ax.set_ylabel("Departure delay (min)")
ax.set_title("Temperature vs delay")
ax.legend(fontsize=9)

# 3b: Precipitation vs delay
ax = axes[0, 1]
merged['prcp_bin'] = pd.cut(merged['prcp'], bins=[-1,0,2,10,1000],
                             labels=['None\n(0mm)','Light\n(0-2mm)','Moderate\n(2-10mm)','Heavy\n(>10mm)'])
prcp_stats = merged.groupby('prcp_bin', observed=True)['delay_min'].agg(['mean','sem','count'])
bars = ax.bar(prcp_stats.index, prcp_stats['mean'],
              color=[TEAL, BLUE, AMBER, CORAL], width=0.6,
              yerr=prcp_stats['sem']*1.96, capsize=5,
              error_kw={'linewidth':1.2})
ax.set_ylabel("Avg departure delay (min)")
ax.set_title("Precipitation level vs delay")
for bar, (_, row) in zip(bars, prcp_stats.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{row['mean']:.1f} min\n(n={int(row['count']):,})",
            ha='center', fontsize=9)

# 3c: Wind speed vs delay
ax = axes[0, 2]
merged['wspd_bin'] = pd.cut(merged['wspd'], bins=8)
wspd_stats = merged.groupby('wspd_bin', observed=True)['delay_min'].agg(['mean','sem']).reset_index()
xvals_w = [b.mid for b in wspd_stats['wspd_bin']]
ax.bar(range(len(wspd_stats)), wspd_stats['mean'], color=AMBER, alpha=0.8,
       yerr=wspd_stats['sem']*1.96, capsize=4,
       error_kw={'linewidth':1.2})
ax.set_xticks(range(len(wspd_stats)))
ax.set_xticklabels([f"{v:.0f}" for v in xvals_w], fontsize=9)
ax.set_xlabel("Wind speed (km/h)")
ax.set_ylabel("Avg departure delay (min)")
ax.set_title("Wind speed vs delay")

# 3d: Pressure vs delay
ax = axes[1, 0]
merged['pres_bin'] = pd.cut(merged['pres'], bins=10)
pres_stats = merged.groupby('pres_bin', observed=True)['delay_min'].agg(['mean','sem']).reset_index()
xvals_p = [b.mid for b in pres_stats['pres_bin']]
ax.plot(xvals_p, pres_stats['mean'], color=CORAL, linewidth=2.2,
        marker='o', markersize=6)
ax.fill_between(xvals_p,
                pres_stats['mean'] - pres_stats['sem']*1.96,
                pres_stats['mean'] + pres_stats['sem']*1.96,
                alpha=0.15, color=CORAL)
ax.set_xlabel("Atmospheric pressure (hPa)")
ax.set_ylabel("Avg departure delay (min)")
ax.set_title("Pressure vs delay\n(lower pressure = stormier conditions)")

# 3e: Temp x Precip heatmap
ax = axes[1, 1]
merged['temp_cat'] = pd.cut(merged['tavg'],
                             bins=[-30,0,10,20,30,50],
                             labels=['<0°C','0–10°C','10–20°C','20–30°C','>30°C'])
merged['prcp_cat'] = pd.cut(merged['prcp'],
                             bins=[-1,0,2,10,1000],
                             labels=['None','Light','Moderate','Heavy'])
heat = merged.pivot_table(values='delay_min', index='temp_cat',
                           columns='prcp_cat', aggfunc='mean')
im = ax.imshow(heat.values, aspect='auto', cmap='YlOrRd', vmin=15, vmax=80)
ax.set_xticks(range(len(heat.columns)))
ax.set_xticklabels(heat.columns, fontsize=10)
ax.set_yticks(range(len(heat.index)))
ax.set_yticklabels(heat.index, fontsize=10)
ax.set_xlabel("Precipitation level")
ax.set_ylabel("Temperature")
ax.set_title("Avg delay: temperature × precipitation (min)")
plt.colorbar(im, ax=ax, label='Avg delay (min)', shrink=0.85)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        val = heat.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.0f}", ha='center', va='center',
                    fontsize=10, color='white' if val > 50 else 'black',
                    fontweight='bold')

# 3f: Delay rate by weather severity
ax = axes[1, 2]
merged['weather_severity'] = 'Mild'
merged.loc[(merged['prcp'] > 2) | (merged['tavg'] < 0) | (merged['tavg'] > 30), 'weather_severity'] = 'Moderate'
merged.loc[(merged['prcp'] > 10) | (merged['snow'] > 0), 'weather_severity'] = 'Severe'
sev_order = ['Mild','Moderate','Severe']
sev_stats = merged.groupby('weather_severity').agg(
    delay_rate=('delay_flag','mean'),
    avg_delay=('delay_min','mean'),
    count=('delay_min','count')
).reindex(sev_order)
x = range(len(sev_order))
width = 0.38
bars1 = ax.bar([i - width/2 for i in x], sev_stats['delay_rate']*100,
               width=width, color=BLUE, alpha=0.85, label='% delayed >15 min')
ax2b = ax.twinx()
bars2 = ax2b.bar([i + width/2 for i in x], sev_stats['avg_delay'],
                 width=width, color=CORAL, alpha=0.85, label='Avg delay (min)')
ax.set_xticks(x)
ax.set_xticklabels(sev_order)
ax.set_ylabel("% flights delayed >15 min", color=BLUE)
ax2b.set_ylabel("Avg departure delay (min)", color=CORAL)
ax.tick_params(axis='y', labelcolor=BLUE)
ax2b.tick_params(axis='y', labelcolor=CORAL)
ax.set_title("Delay rate and magnitude\nby weather severity")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
for bar, val in zip(bars1, sev_stats['delay_rate']*100):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha='center', fontsize=9, color=BLUE)
for bar, val in zip(bars2, sev_stats['avg_delay']):
    ax2b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
              f"{val:.1f}", ha='center', fontsize=9, color=CORAL)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('./IA3_weather_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved IA3_weather_analysis.png")

print("\nAll initial analysis charts saved.")

# The use of GEN-AI was used to assist in coding this analysis section, especially in the design and styling of the charts. The code was then manually reviewed, tested, and refined to ensure correctness and clarity.