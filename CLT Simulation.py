import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, norm, shapiro
import os

#Create figures directory
os.makedirs("figures", exist_ok=True)

np.random.seed(42)

#Step 1 — Implement simulate_clt
def simulate_clt(sample_size, n_simulations=10_000):
    """
    Draw n_simulations × sample_size exponential values from Exp(scale=1.0),
    reshape into (n_simulations, sample_size), and return sample means.
    Vectorized: no explicit loops.
    """
    # Draw all samples at once: shape (n_simulations, sample_size)
    samples = np.random.exponential(scale=1.0, size=(n_simulations, sample_size))
    
    # Return array of sample means (mean over axis=1)
    return samples.mean(axis=1)

#Step 2 — Plot for n ∈ {1, 5, 30}
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, n in zip(axes, [1, 5, 30]):
    means = simulate_clt(sample_size=n, n_simulations=10_000)
    
    # Histogram of sample means
    ax.hist(means, bins=50, density=True, alpha=0.6, label="Sample means")
    
    # Overlay Normal PDF: mean = 1.0, std = 1.0 / sqrt(n)
    x = np.linspace(means.min(), means.max(), 400)
    pdf = norm.pdf(x, loc=1.0, scale=1.0 / np.sqrt(n))
    ax.plot(x, pdf, 'r-', lw=2, label="CLT Normal")
    
    # Observed mean and std
    obs_mean = means.mean()
    obs_std = means.std()
    
    ax.set_title(f"n = {n}\nObserved: mean={obs_mean:.3f}, std={obs_std:.3f}")
    ax.set_xlabel("Sample mean")
    ax.set_ylabel("Density")
    ax.legend()

# Save the figure to figures/clt_panels.png
plt.tight_layout()
plt.savefig("figures/clt_panels.png", dpi=300, bbox_inches="tight")
plt.show()

#Step 3 — Compute the Shapiro-Wilk test
from scipy.stats import shapiro

for n in [1, 5, 30, 100]:
    p = shapiro(simulate_clt(sample_size=n, n_simulations=5_000)).pvalue
    print(f"n={n}: p={p:.4f}")
    
#Step 4 — Writeup
"""
At what sample size does the distribution of sample means start looking Normal?
From the histograms, the distribution of sample means looks roughly bell-shaped by n = 5, but it clearly matches the Normal curve (symmetric, smooth, no heavy skew) by n = 30. For an exponential source, n ≈ 30 is where the CLT approximation becomes visually convincing.

The Exponential distribution has mean = 1.0 and std = 1.0. Why does the spread of the sample-mean distribution shrink as 
n grows? Give the exact formula.
The spread shrinks because the standard deviation of the sample mean is the population standard deviation divided by n. For an exponential with 
σ=1, the standard error of the mean is:  SE(𝑋)=𝜎/√𝑛.
As n increases, this quantity decreases, so the histogram of sample means becomes narrower.

The CLT says “sample mean is approximately Normal”. For a skewed source like Exponential, how large does 
n typically need to be in practice for the approximation to be safe?
In practice, for moderately skewed distributions like the exponential, n ≈ 30 is often considered large enough for the Normal approximation to be reasonably accurate for many purposes (confidence intervals, hypothesis tests). For more skewed or heavy-tailed distributions, larger 
n (e.g., 50–100+) may be needed for high precision, but for exponential data, n = 30 is typically sufficient.
"""