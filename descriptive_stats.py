import pandas as pd
import matplotlib.pyplot as plt
import statistics
from scipy.stats import skew, kurtosis
import seaborn as sns
import os

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)

df = pd.read_csv("lumen_users.csv")
print("working")
column_headers = list(df)
print("Columns: ", list(df), "\n")

# Q1 – Histogram
fig1 = plt.figure(figsize=(10, 6))
plt.hist(df["monthly_watch_hours"], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Monthly Watch Hours')
plt.xlabel('Monthly Watch Hours')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("figures/watch_hours_hist.png", dpi=300, bbox_inches='tight')
plt.show()

# Q2 – Summary by plan (unchanged logic, just print)
summary_table = df.groupby('plan', as_index=False).agg(
    N_USERS=('user_id', 'size'),
    MEAN=('monthly_watch_hours', 'mean'),
    MEDIAN=('monthly_watch_hours', 'median')
)
summary_table['MEAN - MEDIAN'] = summary_table['MEAN'] - summary_table['MEDIAN']
print(summary_table)
print("Q2:  Premium plan shows the largest gap between mean and median showing highest engagement variability\n")

# Q3 – Percentiles by plan and top 1% multiplier (unchanged)
summary_table = df.groupby('plan', as_index=False).agg(
    N_users=('user_id', 'size'),
    P05=('monthly_watch_hours', lambda x: x.quantile(0.05)),
    P25=('monthly_watch_hours', lambda x: x.quantile(0.25)),
    P50=('monthly_watch_hours', lambda x: x.quantile(0.50)),
    P75=('monthly_watch_hours', lambda x: x.quantile(0.75)),
    P95=('monthly_watch_hours', lambda x: x.quantile(0.95)),
    P99=('monthly_watch_hours', lambda x: x.quantile(0.99))
)

p99_threshold = df['monthly_watch_hours'].quantile(0.99)
median_val = df['monthly_watch_hours'].quantile(0.50)

top_1_percent_users = df[df['monthly_watch_hours'] > p99_threshold]
top_1_mean = top_1_percent_users['monthly_watch_hours'].mean()
true_multiplier = top_1_mean / median_val

print(f"\nMedian User Watch Time: {median_val:.2f} hours")
print(f"P99 Threshold (Entry to Top 1%): {p99_threshold:.2f} hours")
print(f"True Average Top 1% User: {top_1_mean:.2f} hours")
print(f"Q3:  TRUE MULTIPLIER (Top 1% Avg / Median): {true_multiplier:.2f}x")   

# Q4 – IQR outliers (unchanged)
Q1 = df['monthly_watch_hours'].quantile(0.25)
Q3 = df['monthly_watch_hours'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Clamp lower bound to 0 if negative
lower_bound = max(0, lower_bound)

outlier_mask = (df['monthly_watch_hours'] < lower_bound) | (df['monthly_watch_hours'] > upper_bound)

n_outliers = outlier_mask.sum()
total_rows = len(df)
outlier_percentage_count = (n_outliers / total_rows) * 100

total_watch_hours = df['monthly_watch_hours'].sum()
outlier_watch_hours = df.loc[outlier_mask, 'monthly_watch_hours'].sum()
outlier_percentage_hours = (outlier_watch_hours / total_watch_hours) * 100

print(f"\nTotal Rows: {total_rows}")
print(f"Outlier Count: {n_outliers} ({outlier_percentage_count:.2f}% of rows)")
print(f"Total Watch Hours: {total_watch_hours:.2f}")
print(f"Q4: Outlier Watch Hours: {outlier_watch_hours:.2f}")
print(f"**Outliers account for {outlier_percentage_hours:.2f}% of total watch hours.**")

# Boxplot by plan (Q4 continued)
fig2 = plt.figure(figsize=(10, 6))
sns.boxplot(x='plan', y='monthly_watch_hours', data=df, showfliers=True)
plt.title('Distribution of Monthly Watch Hours by Plan')
plt.xlabel('Plan')
plt.ylabel('Monthly Watch Hours')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("figures/watch_hours_by_plan.png", dpi=300, bbox_inches='tight')
plt.show()

# Q5 – Narrative
print("The typical user watches 6.6hrs/month with most users between a range of 3.64-12.35 hrs/month")
print("\n script finished")