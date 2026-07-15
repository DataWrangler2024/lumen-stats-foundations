import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t


# -----------------------
# Load data
# -----------------------
df = pd.read_csv(
    r"C:\Users\keith.frost\Documents\Python\Data Scientist D8a.academy\Descriptive Stats\lumen_users.csv"
)


# -----------------------
# Confidence interval for mean (parametric, t-based)
# -----------------------
def ci_mean(data, alpha=0.05):
    data = pd.Series(data).dropna()
    n = len(data)
    mean = data.mean()
    std_err = data.std(ddof=1) / np.sqrt(n)
    margin = t.ppf(1 - alpha / 2, df=n - 1) * std_err
    return mean, mean - margin, mean + margin


# -----------------------
# Bootstrap confidence interval
# -----------------------
def bootstrap_ci(data, statistic=np.mean, alpha=0.05, B=10_000, seed=42):
    data = np.asarray(pd.Series(data).dropna())
    replicates = np.empty(B)
    rng = np.random.default_rng(seed=seed)
    n = len(data)
    for i in range(B):
        sample = rng.choice(data, size=n, replace=True)
        replicates[i] = statistic(sample)
    point = statistic(data)
    low = np.percentile(replicates, 100 * alpha / 2)
    high = np.percentile(replicates, 100 * (1 - alpha / 2))
    return point, low, high


# -----------------------
# Step 1 — Parametric CI for FR vs US
# -----------------------
fr_users = df.loc[df["country"] == "FR", "monthly_watch_hours"]
us_users = df.loc[df["country"] == "US", "monthly_watch_hours"]


fr_mean, fr_low, fr_high = ci_mean(fr_users)
us_mean, us_low, us_high = ci_mean(us_users)


parametric_table = pd.DataFrame({
    "country": ["FR", "US"],
    "mean": [fr_mean, us_mean],
    "95% CI (low, high)": [(fr_low, fr_high), (us_low, us_high)]
})


print("Step 1 — Parametric CI")
print(parametric_table.to_string(index=False))
print("Overlap:", not (fr_high < us_low or us_high < fr_low))


# -----------------------
# Step 2 — Bootstrap CI for FR vs US
# -----------------------
fr_boot_mean = bootstrap_ci(fr_users, statistic=np.mean)
us_boot_mean = bootstrap_ci(us_users, statistic=np.mean)


bootstrap_table = pd.DataFrame({
    "country": ["FR", "US"],
    "mean": [fr_boot_mean[0], us_boot_mean[0]],
    "95% CI (low, high)": [
        (fr_boot_mean[1], fr_boot_mean[2]),
        (us_boot_mean[1], us_boot_mean[2])
    ]
})


print("\nStep 2 — Bootstrap CI")
print(bootstrap_table.to_string(index=False))


# -----------------------
# Step 3 — Bootstrap for other statistics (FR only)
# -----------------------
mean_ci = bootstrap_ci(fr_users, statistic=np.mean)
median_ci = bootstrap_ci(fr_users, statistic=np.median)
p90_ci = bootstrap_ci(fr_users, statistic=lambda x: np.percentile(x, 90))


bootstrap_stats = pd.DataFrame({
    "statistic": ["mean", "median", "p90"],
    "point_estimate": [mean_ci[0], median_ci[0], p90_ci[0]],
    "95% CI low": [mean_ci[1], median_ci[1], p90_ci[1]],
    "95% CI high": [mean_ci[2], median_ci[2], p90_ci[2]],
})


print("\nStep 3 — Bootstrap stats for FR")
print(bootstrap_stats.to_string(index=False))


# -----------------------
# Step 4 — Plot by country with 95% CI
# -----------------------
countries = df.groupby("country").filter(lambda g: len(g) >= 100)


def country_ci_stats(group):
    data = group["monthly_watch_hours"].dropna()
    n = len(data)
    if n == 0:
        return pd.Series({"mean": np.nan, "low": np.nan, "high": np.nan})
    mean = data.mean()
    std_err = data.std(ddof=1) / np.sqrt(n)
    margin = t.ppf(1 - 0.05 / 2, df=n - 1) * std_err
    return pd.Series({
        "mean": mean,
        "low": mean - margin,
        "high": mean + margin
    })


# Apply per country group, excluding the grouping column from the input
agg = countries.groupby("country")[["monthly_watch_hours"]].apply(country_ci_stats, include_groups=False)

# Ensure agg is a DataFrame and has the right columns
agg = agg[["mean", "low", "high"]]
agg = agg.sort_values("mean")


os.makedirs("C:\\Users\\keith.frost\\Documents\\Python\\Data Scientist D8a.academy\\Confidence Intervals\\figures", exist_ok=True)

plt.figure(figsize=(10, 6))
plt.errorbar(
    agg["mean"].index,
    agg["mean"],
    yerr=[agg["mean"] - agg["low"], agg["high"] - agg["mean"]],
    fmt="o",
    ecolor="black",
    capsize=5,
    elinewidth=1.5,
    markersize=6
)

plt.xticks(rotation=45, ha="right")
plt.xlabel("Country")
plt.ylabel("Mean monthly_watch_hours")
plt.title("Mean Monthly Watch Hours by Country (≥100 users) with 95% CI")

plt.tight_layout()
plt.savefig("C:\\Users\\keith.frost\\Documents\\Python\\Data Scientist D8a.academy\\Confidence Intervals\\figures\\watch_by_country_ci.png", dpi=150)
plt.close()
print("\nStep 4 — Plot saved to figures/watch_by_country_ci.png")


# -----------------------
# Step 5 — PM-style response
# -----------------------
fr_us_overlap = not (fr_high < us_low or us_high < fr_low)
overlap_amount = min(fr_high, us_high) - max(fr_low, us_low)

pm_reply = f"""
French users have a higher mean watch time than US users, and the 95% CIs {'do' if fr_us_overlap else 'do not'} overlap.
The overlap is {overlap_amount:.2f} hours, so the lift is {'suggestive but not definitive' if fr_us_overlap else 'strongly supported'}.
For the country view, the clearest standouts are the countries with means farthest from the pack and intervals that separate most clearly from the rest.
"""


print("\nStep 5 — PM reply")
print(pm_reply.strip())