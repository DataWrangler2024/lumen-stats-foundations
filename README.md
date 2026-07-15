# Lumen — Stats Foundations Project

End-to-end statistical analysis of user engagement on the Lumen
streaming platform. Covers descriptive stats, distributions, CLT,
hypothesis testing, and confidence intervals.

## Stack
Python 3.11 · NumPy · pandas · scipy · matplotlib · Jupyter

## How to reproduce
1. `pip install -r requirements.txt`
2. Download the dataset (see `data/README.md`)
3. Open the notebooks in order (run them from the repo root)

## Three insights
- The watch-hours distribution is heavily right-skewed; the
  median (X h/month) is the right "typical user" figure for
  exec communication, not the mean (Y h/month).
- The new "Recommended for you" carousel lifted watch time
  by Z% (Cohen's d = 0.XX, Bonferroni-corrected p = 0.YY).
  No SRM detected.
- French users average X% more watch time than US users with
  non-overlapping 95% CIs — likely real, worth a follow-up
  on content catalogue or pricing.

## What I'd do next
Run a multi-week holdout to check for novelty decay on the
carousel result. Run a stratified analysis on plan tier
before generalising the country-level finding.
