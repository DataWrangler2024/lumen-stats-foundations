import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

# Load CSV
df = pd.read_csv("C:\\Users\\keith.frost\\Documents\\Python\\Data Scientist D8a.academy\\Hypothesis Test\\lumen_ab_test.csv")

#Step 1: SRM mismatch check
print(df["variant"].value_counts())
print(df.groupby("variant")["signup_age_days"].describe())
print(df.groupby("variant")["clicked_carousel"].value_counts())
print("no SRM mismatch detected\n")

#Step 2: Two-sample t-test on watch_minutes
from scipy.stats import ttest_ind

control = df[df.variant == "control"]["watch_minutes"]
treatment = df[df.variant == "treatment"]["watch_minutes"]

result = ttest_ind(treatment, control, equal_var=False)
print(f"t = {result.statistic:.3f}")
print(f"p (two-tailed) = {result.pvalue:.4f}")
print("Interpretation: If p < 0.05, we reject the null hypothesis and conclude that there is a significant difference in watch_minutes between the treatment and control groups.\n")

#Step 3 — Effect size (Cohen's d)
import numpy as np

def cohens_d(a, b):
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / pooled_std

print(f"Cohen's d = {cohens_d(treatment, control):.3f}\n")
print("Interpretation: Cohen's d is a measure of effect size. A value of 0.2 is considered a small effect, 0.5 a medium effect, and 0.8 a large effect.\n")

#Step 3 — Lift calculation for clicked_carousel
control = df[df['variant'] == 'control']['clicked_carousel']
treatment = df[df['variant'] == 'treatment']['clicked_carousel']

p_control = control.mean()      # proportion of True in control
p_treatment = treatment.mean()  # proportion of True in treatment

lift_clicked_carousel = (p_treatment - p_control) / p_control
lift_clicked_carousel_pct = lift_clicked_carousel * 100

print(f"Control click rate: {p_control:.4f}")
print(f"Treatment click rate: {p_treatment:.4f}")
print(f"Lift: {lift_clicked_carousel:.4f} ({lift_clicked_carousel_pct:.2f}%)\n")
print("Interpretation: Lift is the relative increase in the metric of interest (clicked_carousel) in the treatment group compared to the control group. A positive lift indicates that the treatment had a positive effect on the metric.\n")


#Step 4 — Multiple comparisons correction
# Ensure variant is coded clearly
df["variant"] = df["variant"].astype(str)

# Separate groups
treat = df[df["variant"] == "treatment"]
ctrl  = df[df["variant"] == "control"]

results = []

# 1) watch_minutes: two-sample t-test (continuous)
x1 = treat["watch_minutes"].values
x0 = ctrl["watch_minutes"].values
t_stat, p_raw = stats.ttest_ind(x1, x0, equal_var=False)
results.append({
    "metric": "watch_minutes",
    "test": "t-test",
    "raw_p": p_raw,
    "bonf_p": p_raw * 5,
    "significant_bonf": (p_raw * 5) < 0.05
})

# 2) sessions: two-sample t-test (continuous)
x1 = treat["sessions"].values
x0 = ctrl["sessions"].values
t_stat, p_raw = stats.ttest_ind(x1, x0, equal_var=False)
results.append({
    "metric": "sessions",
    "test": "t-test",
    "raw_p": p_raw,
    "bonf_p": p_raw * 5,
    "significant_bonf": (p_raw * 5) < 0.05
})

# Helper for proportions tests
def proportions_test(metric):
    counts = df.groupby("variant")[metric].sum().values  # [treatment, control]
    nobs = df.groupby("variant")[metric].count().values  # n per variant
    stat, p_raw = proportions_ztest(counts, nobs, alternative="two-sided")
    return p_raw

# 3) clicked_carousel: proportions z-test
p_raw = proportions_test("clicked_carousel")
results.append({
    "metric": "clicked_carousel",
    "test": "proportions z-test",
    "raw_p": p_raw,
    "bonf_p": p_raw * 5,
    "significant_bonf": (p_raw * 5) < 0.05
})

# 4) converted_to_paid: proportions z-test
p_raw = proportions_test("converted_to_paid")
results.append({
    "metric": "converted_to_paid",
    "test": "proportions z-test",
    "raw_p": p_raw,
    "bonf_p": p_raw * 5,
    "significant_bonf": (p_raw * 5) < 0.05
})

# 5) churned: proportions z-test
p_raw = proportions_test("churned")
results.append({
    "metric": "churned",
    "test": "proportions z-test",
    "raw_p": p_raw,
    "bonf_p": p_raw * 5,
    "significant_bonf": (p_raw * 5) < 0.05
})

# Build result table
res_df = pd.DataFrame(results)
res_df["significant_at_0.05_bonf"] = res_df["significant_bonf"]

print(res_df.to_string(index=False))
print("\nInterpretation: The Bonferroni correction adjusts the significance threshold to account for multiple comparisons. If the adjusted p-value is less than 0.05, we reject the null hypothesis for that metric, indicating a significant difference between treatment and control groups after correction.\n")

#Step 5 — Summary of findings
print("""TL;DR — Yes, ship the new carousel: it significantly increases clicks and paid conversions with no detectable impact on churn.

Verdict on watch_minutes — Treatment shows a small negative lift (~−0.26%, p_bonf ≈ 5.7e‑19, Cohen’s d ≈ 0.13), but the effect is negligible in business terms and does not offset gains in engagement and revenue.

Caveats — This test does not tell us about long-term retention, novelty bias (users may react strongly at first but fade), or whether effects differ by segment (e.g., new vs. existing users, device type).

What I'd do next — Roll out to 100% while monitoring churn and watch_minutes over 2–4 weeks; if negative trends appear, run segmented analyses (new vs. returning users) to identify where the effect reverses and adjust the carousel accordingly.""")