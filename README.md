# 📊 Causal Impact of Gig Worker Reclassification (Prop 22)

An end-to-end causal inference pipeline estimating the effect of California's Proposition 22 on gig worker employment, earnings, and consumer prices using Difference-in-Differences and Synthetic Control methods.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14+-green)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

California's Proposition 22 (2020) reclassified gig workers at app-based companies as independent contractors rather than employees. This project builds a complete causal inference pipeline to estimate the policy's effects on three outcomes:

1. **Gig employment**: Did the policy increase or decrease the number of gig workers?
2. **Worker earnings**: How were monthly earnings affected?
3. **Consumer prices**: Did ride prices change for consumers?

### Methods

| Method | Purpose |
|--------|---------|
| Two-Way Fixed Effects DID | Primary treatment effect estimation |
| Event Study | Testing parallel trends assumption |
| Synthetic Control | Alternative estimator using donor pool |
| Robustness Battery | Covariate adjustment, trimming, placebo tests, alternative clustering |

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/causal-impact-prop22.git
cd causal-impact-prop22
pip install -r requirements.txt
python analysis.py
```

### Generate Publication Figures

```bash
python visualize.py
```

Figures are saved to `figures/`.

## Pipeline Architecture

```
1. Data Generation     → Synthetic panel calibrated to BLS/Census patterns
2. DID Estimation      → TWFE with clustered SEs
3. Event Study         → Dynamic treatment effects + parallel trends F-test
4. Robustness Checks   → 5 specifications (basic, covariates, alt. clustering,
                          trimmed sample, placebo test)
5. Synthetic Control   → Donor pool optimization + in-space placebo tests
6. Visualization       → Publication-quality figures
```

## Key Design Decisions

**Why synthetic data?** Real BLS QCEW microdata requires restricted access. This pipeline uses synthetic data calibrated to published summary statistics and known effect sizes, allowing full reproducibility. The code is structured so that replacing the data generator with real data loading requires changing only one function.

**Why both DID and SC?** DID relies on the parallel trends assumption (testable). Synthetic Control constructs an explicit counterfactual. When both methods agree, the result is more credible. When they disagree, it reveals important identification concerns.

**True effects embedded in the data:**
- Employment: +8% (positive, as independent contractor status reduces hiring friction)
- Earnings: -4% (negative, as benefit obligations shift to workers)
- Prices: -5% (negative, as lower labor costs pass through to consumers)

## Output

The analysis prints a structured summary:

```
SUMMARY OF RESULTS
======================================================================
         Outcome    DID Estimate      SE  p-value  SC Estimate  True Effect
  Gig Employment         0.0823  0.0098   0.0000       0.0791         0.08
Monthly Earnings        -0.0387  0.0071   0.0000      -0.0412        -0.04
  Avg Ride Price        -0.0512  0.0053   0.0000      -0.0489        -0.05
```

## Project Structure

```
causal-impact-prop22/
├── analysis.py          # Core pipeline (data gen, DID, SC, robustness)
├── visualize.py         # Publication-quality figures
├── requirements.txt     # Dependencies
├── README.md            # This file
└── figures/             # Generated output
    ├── event_study_*.png
    ├── synthetic_control_*.png
    ├── parallel_trends_*.png
    └── robustness_*.png
```

## Relevance

This project demonstrates the complete toolkit expected of a tech economist:

- **Causal inference**: DID, event study, synthetic control (the core of tech economist interviews)
- **Econometric rigor**: Clustered SEs, parallel trends testing, placebo tests, robustness battery
- **Software engineering**: OOP design, type hints, docstrings, modular architecture
- **Communication**: Clear output, publication figures, documented methodology

## Extending This Project

To use with real data, replace `generate_panel_data()` with a function that loads your actual panel dataset. The rest of the pipeline works without modification as long as the DataFrame has the expected columns (`state`, `month`, `treated`, `post`, `did`, and outcome variables).

## References

- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic Control Methods for Comparative Case Studies. JASA.
- Angrist, J. D., & Pischke, J. S. (2009). Mostly Harmless Econometrics.
- California Secretary of State. Proposition 22 (2020).

## License

MIT License.
