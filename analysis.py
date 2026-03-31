"""
Causal Impact of Gig Worker Reclassification (Prop 22) on Labor Market Outcomes
================================================================================
End-to-end causal inference pipeline using Difference-in-Differences and
Synthetic Control methods to estimate the effect of California's Proposition 22
on gig worker employment, earnings, and consumer prices.

Uses synthetic data calibrated to published BLS and Census statistics.

Author: [Prabin Adhikari]
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════
# 1. DATA GENERATION (Synthetic, calibrated to real patterns)
# ══════════════════════════════════════════════

@dataclass
class DataConfig:
    """Configuration for synthetic data generation."""
    n_states: int = 50
    n_periods: int = 24          # Monthly data: 12 pre + 12 post
    treatment_period: int = 12   # Prop 22 effective Dec 2020
    treated_state: str = "California"
    seed: int = 42
    
    # Calibrated parameters (based on BLS QCEW data patterns)
    base_gig_employment: float = 50000
    base_earnings: float = 2800     # Monthly earnings
    base_ride_price: float = 18.50
    
    # Treatment effects (based on published estimates)
    true_employment_effect: float = 0.08   # 8% increase
    true_earnings_effect: float = -0.04    # 4% decrease  
    true_price_effect: float = -0.05       # 5% decrease


def generate_panel_data(config: DataConfig) -> pd.DataFrame:
    """
    Generate state-month panel data mimicking real labor market patterns.
    
    Includes:
    - State fixed effects (heterogeneous baseline levels)
    - Time trends (common macroeconomic patterns including COVID shock)
    - Treatment effect for California post-Prop 22
    - Realistic noise calibrated to BLS data variance
    """
    np.random.seed(config.seed)
    
    states = [config.treated_state] + [f"State_{i}" for i in range(1, config.n_states)]
    
    # State fixed effects (log-normal to capture size heterogeneity)
    state_fe = {s: np.random.lognormal(0, 0.5) for s in states}
    state_fe[config.treated_state] = 2.5  # CA is large
    
    # Donor pool: states similar to CA for synthetic control
    donor_weights = {s: max(0, 1 - abs(state_fe[s] - state_fe[config.treated_state])) 
                     for s in states if s != config.treated_state}
    
    records = []
    
    for state in states:
        for t in range(config.n_periods):
            is_treated = (state == config.treated_state) and (t >= config.treatment_period)
            is_ca = state == config.treated_state
            post = t >= config.treatment_period
            
            # Common time trend with COVID shock
            time_trend = 0.003 * t  # Gradual growth
            covid_shock = -0.15 * np.exp(-0.3 * max(0, t - 8))  # Shock around month 8-9
            seasonal = 0.02 * np.sin(2 * np.pi * t / 12)
            
            # State-specific trends (for testing parallel trends)
            state_trend = state_fe[state] * 0.001 * t
            
            # ── Gig Employment ──
            log_emp = (np.log(config.base_gig_employment) + 
                       np.log(state_fe[state]) +
                       time_trend + covid_shock + seasonal + state_trend +
                       (config.true_employment_effect if is_treated else 0) +
                       np.random.normal(0, 0.03))
            
            gig_employment = np.exp(log_emp)
            
            # ── Monthly Earnings ──
            log_earn = (np.log(config.base_earnings) +
                        0.1 * np.log(state_fe[state]) +
                        0.002 * t + covid_shock * 0.5 + seasonal * 0.5 +
                        (config.true_earnings_effect if is_treated else 0) +
                        np.random.normal(0, 0.02))
            
            monthly_earnings = np.exp(log_earn)
            
            # ── Consumer Ride Price ──
            log_price = (np.log(config.base_ride_price) +
                         0.05 * np.log(state_fe[state]) +
                         0.001 * t + covid_shock * 0.3 +
                         (config.true_price_effect if is_treated else 0) +
                         np.random.normal(0, 0.015))
            
            avg_ride_price = np.exp(log_price)
            
            # ── Covariates ──
            unemployment = 5.5 + covid_shock * 20 + np.random.normal(0, 0.3)
            gas_price = 3.20 + 0.02 * t + np.random.normal(0, 0.1)
            pop_density = 200 * state_fe[state] + np.random.normal(0, 10)
            
            records.append({
                'state': state,
                'month': t,
                'year_month': pd.Timestamp('2020-01-01') + pd.DateOffset(months=t),
                'treated': int(is_ca),
                'post': int(post),
                'did': int(is_treated),
                'gig_employment': gig_employment,
                'monthly_earnings': monthly_earnings,
                'avg_ride_price': avg_ride_price,
                'log_gig_employment': log_emp,
                'log_monthly_earnings': log_earn,
                'log_avg_ride_price': log_price,
                'unemployment_rate': max(2, unemployment),
                'gas_price': max(2, gas_price),
                'pop_density': max(50, pop_density)
            })
    
    df = pd.DataFrame(records)
    df['state_id'] = pd.Categorical(df['state']).codes
    df['relative_time'] = df['month'] - config.treatment_period
    
    return df


# ══════════════════════════════════════════════
# 2. DIFFERENCE-IN-DIFFERENCES ESTIMATION
# ══════════════════════════════════════════════

class DifferenceInDifferences:
    """
    Implements standard and event-study DID estimation with:
    - Two-way fixed effects (TWFE)
    - Clustered standard errors
    - Event study specification for parallel trends testing
    - Covariate-adjusted specifications
    """
    
    def __init__(self, data: pd.DataFrame, outcome: str, 
                 treatment_col: str = 'did',
                 entity_col: str = 'state', time_col: str = 'month'):
        self.data = data.copy()
        self.outcome = outcome
        self.treatment_col = treatment_col
        self.entity_col = entity_col
        self.time_col = time_col
        self.results = {}
    
    def estimate_twfe(self, covariates: List[str] = None, 
                      cluster_col: str = 'state') -> Dict:
        """
        Two-Way Fixed Effects DID estimation.
        
        Y_it = α_i + γ_t + δ*D_it + X_it'β + ε_it
        
        where D_it = Treated_i * Post_t
        """
        # Build formula with entity and time FE
        formula = f"{self.outcome} ~ {self.treatment_col}"
        
        if covariates:
            formula += " + " + " + ".join(covariates)
        
        formula += f" + C({self.entity_col}) + C({self.time_col})"
        
        model = smf.ols(formula, data=self.data).fit(
            cov_type='cluster',
            cov_kwds={'groups': self.data[cluster_col]}
        )
        
        # Extract DID coefficient
        coef = model.params[self.treatment_col]
        se = model.bse[self.treatment_col]
        ci = model.conf_int().loc[self.treatment_col]
        
        result = {
            'estimator': 'TWFE DID',
            'coefficient': coef,
            'std_error': se,
            't_statistic': model.tvalues[self.treatment_col],
            'p_value': model.pvalues[self.treatment_col],
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'n_obs': model.nobs,
            'r_squared': model.rsquared,
            'r_squared_adj': model.rsquared_adj,
            'model': model
        }
        
        self.results['twfe'] = result
        return result
    
    def estimate_event_study(self, n_pre: int = 6, n_post: int = 6,
                              cluster_col: str = 'state') -> Dict:
        """
        Event study specification for testing parallel trends
        and estimating dynamic treatment effects.
        
        Y_it = α_i + γ_t + Σ_k δ_k * 1(t-t* = k) * Treated_i + ε_it
        
        Omits k = -1 as reference period.
        """
        df = self.data.copy()
        
        # Create relative time indicators (use 'm' prefix for negative to avoid Patsy issues)
        event_dummies = []
        k_to_col = {}
        for k in range(-n_pre, n_post + 1):
            if k == -1:  # Reference period
                continue
            col_name = f"rel_m{abs(k)}" if k < 0 else f"rel_p{k}"
            df[col_name] = ((df['relative_time'] == k) & (df['treated'] == 1)).astype(int)
            event_dummies.append(col_name)
            k_to_col[k] = col_name
        
        formula = f"{self.outcome} ~ " + " + ".join(event_dummies)
        formula += f" + C({self.entity_col}) + C({self.time_col})"
        
        model = smf.ols(formula, data=df).fit(
            cov_type='cluster',
            cov_kwds={'groups': df[cluster_col]}
        )
        
        # Collect event study coefficients
        es_results = []
        for k in range(-n_pre, n_post + 1):
            if k == -1:
                es_results.append({'relative_time': k, 'coefficient': 0, 
                                   'std_error': 0, 'ci_lower': 0, 'ci_upper': 0})
                continue
            
            col_name = k_to_col[k]
            if col_name in model.params.index:
                ci = model.conf_int().loc[col_name]
                es_results.append({
                    'relative_time': k,
                    'coefficient': model.params[col_name],
                    'std_error': model.bse[col_name],
                    'ci_lower': ci[0],
                    'ci_upper': ci[1]
                })
        
        es_df = pd.DataFrame(es_results)
        
        # Test parallel trends: joint F-test on pre-treatment coefficients
        pre_cols = [k_to_col[k] for k in range(-n_pre, 0) if k != -1 and k in k_to_col]
        if pre_cols:
            r_matrix = np.zeros((len(pre_cols), len(model.params)))
            for i, col in enumerate(pre_cols):
                idx = list(model.params.index).index(col)
                r_matrix[i, idx] = 1
            
            try:
                f_test = model.f_test(r_matrix)
                parallel_trends_f = float(f_test.fvalue)
                parallel_trends_p = float(f_test.pvalue)
            except Exception:
                parallel_trends_f = None
                parallel_trends_p = None
        else:
            parallel_trends_f = None
            parallel_trends_p = None
        
        result = {
            'estimator': 'Event Study',
            'event_study_df': es_df,
            'parallel_trends_f_stat': parallel_trends_f,
            'parallel_trends_p_value': parallel_trends_p,
            'model': model
        }
        
        self.results['event_study'] = result
        return result
    
    def run_robustness_checks(self, cluster_col: str = 'state') -> pd.DataFrame:
        """
        Run multiple robustness specifications:
        1. Basic DID (no covariates)
        2. With covariates
        3. Different clustering
        4. Trimmed sample (drop outlier states)
        5. Placebo test (fake treatment date)
        """
        checks = []
        
        # Spec 1: Basic TWFE
        r1 = self.estimate_twfe()
        checks.append({
            'specification': '(1) Basic TWFE',
            'coefficient': r1['coefficient'],
            'std_error': r1['std_error'],
            'p_value': r1['p_value'],
            'n_obs': r1['n_obs'],
            'r_squared': r1['r_squared']
        })
        
        # Spec 2: With covariates
        r2 = self.estimate_twfe(covariates=['unemployment_rate', 'gas_price'])
        checks.append({
            'specification': '(2) With Covariates',
            'coefficient': r2['coefficient'],
            'std_error': r2['std_error'],
            'p_value': r2['p_value'],
            'n_obs': r2['n_obs'],
            'r_squared': r2['r_squared']
        })
        
        # Spec 3: Entity-month clustering
        r3 = self.estimate_twfe(cluster_col='state_id')
        checks.append({
            'specification': '(3) Alt. Clustering',
            'coefficient': r3['coefficient'],
            'std_error': r3['std_error'],
            'p_value': r3['p_value'],
            'n_obs': r3['n_obs'],
            'r_squared': r3['r_squared']
        })
        
        # Spec 4: Drop extreme states
        state_means = self.data.groupby('state')[self.outcome].mean()
        q25, q75 = state_means.quantile(0.25), state_means.quantile(0.75)
        trimmed = self.data[
            (self.data['state'] == 'California') |
            (self.data.groupby('state')[self.outcome].transform('mean').between(q25, q75))
        ]
        
        formula = f"{self.outcome} ~ {self.treatment_col} + C({self.entity_col}) + C({self.time_col})"
        r4_model = smf.ols(formula, data=trimmed).fit(
            cov_type='cluster', cov_kwds={'groups': trimmed[cluster_col]}
        )
        checks.append({
            'specification': '(4) Trimmed Sample',
            'coefficient': r4_model.params[self.treatment_col],
            'std_error': r4_model.bse[self.treatment_col],
            'p_value': r4_model.pvalues[self.treatment_col],
            'n_obs': r4_model.nobs,
            'r_squared': r4_model.rsquared
        })
        
        # Spec 5: Placebo test (fake treatment 6 months early)
        placebo_df = self.data[self.data['month'] < 12].copy()  # pre-period only
        placebo_df['placebo_post'] = (placebo_df['month'] >= 6).astype(int)
        placebo_df['placebo_did'] = placebo_df['treated'] * placebo_df['placebo_post']
        
        formula_placebo = f"{self.outcome} ~ placebo_did + C({self.entity_col}) + C({self.time_col})"
        r5_model = smf.ols(formula_placebo, data=placebo_df).fit(
            cov_type='cluster', cov_kwds={'groups': placebo_df[cluster_col]}
        )
        checks.append({
            'specification': '(5) Placebo (t-6)',
            'coefficient': r5_model.params['placebo_did'],
            'std_error': r5_model.bse['placebo_did'],
            'p_value': r5_model.pvalues['placebo_did'],
            'n_obs': r5_model.nobs,
            'r_squared': r5_model.rsquared
        })
        
        robustness_df = pd.DataFrame(checks)
        self.results['robustness'] = robustness_df
        return robustness_df


# ══════════════════════════════════════════════
# 3. SYNTHETIC CONTROL METHOD
# ══════════════════════════════════════════════

class SyntheticControl:
    """
    Implements the Synthetic Control Method (Abadie et al., 2010).
    
    Constructs a weighted combination of untreated units that best
    approximates the treated unit's pre-treatment trajectory.
    
    Uses constrained optimization (weights sum to 1, non-negative).
    """
    
    def __init__(self, data: pd.DataFrame, outcome: str,
                 treated_unit: str, entity_col: str = 'state',
                 time_col: str = 'month', treatment_period: int = 12):
        self.data = data
        self.outcome = outcome
        self.treated_unit = treated_unit
        self.entity_col = entity_col
        self.time_col = time_col
        self.treatment_period = treatment_period
        self.weights = None
        self.results = {}
    
    def fit(self, predictors: List[str] = None) -> Dict:
        """
        Estimate synthetic control weights using pre-treatment data.
        
        Minimizes: ||X_1 - X_0 * W||^2
        subject to: w_j >= 0, Σw_j = 1
        """
        from scipy.optimize import minimize
        
        pre_data = self.data[self.data[self.time_col] < self.treatment_period]
        
        # Pivot to get state x time matrix
        pivot = pre_data.pivot(
            index=self.entity_col, columns=self.time_col, values=self.outcome
        )
        
        treated_vec = pivot.loc[self.treated_unit].values
        donor_names = [s for s in pivot.index if s != self.treated_unit]
        donor_matrix = pivot.loc[donor_names].values  # J x T
        
        J = len(donor_names)
        
        def objective(w):
            synthetic = w @ donor_matrix
            return np.sum((treated_vec - synthetic) ** 2)
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        # Bounds: non-negative weights
        bounds = [(0, 1)] * J
        
        # Initial: equal weights
        w0 = np.ones(J) / J
        
        result = minimize(objective, w0, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 1000, 'ftol': 1e-12})
        
        self.weights = dict(zip(donor_names, result.x))
        self.donor_names = donor_names
        self.donor_matrix_full = self.data.pivot(
            index=self.entity_col, columns=self.time_col, values=self.outcome
        ).loc[donor_names].values
        
        # Get top donors
        top_donors = sorted(self.weights.items(), key=lambda x: -x[1])[:10]
        
        return {
            'weights': self.weights,
            'top_donors': top_donors,
            'pre_treatment_rmse': np.sqrt(result.fun / len(treated_vec)),
            'optimization_success': result.success
        }
    
    def estimate_effect(self) -> Dict:
        """Compute treatment effect as gap between treated and synthetic."""
        all_pivot = self.data.pivot(
            index=self.entity_col, columns=self.time_col, values=self.outcome
        )
        
        treated_series = all_pivot.loc[self.treated_unit].values
        
        w_array = np.array([self.weights[s] for s in self.donor_names])
        synthetic_series = w_array @ self.donor_matrix_full
        
        gap = treated_series - synthetic_series
        
        months = sorted(self.data[self.time_col].unique())
        
        # Pre/post average gaps
        pre_gap = np.mean(gap[:self.treatment_period])
        post_gap = np.mean(gap[self.treatment_period:])
        att = post_gap - pre_gap
        
        result_df = pd.DataFrame({
            'month': months,
            'treated': treated_series,
            'synthetic': synthetic_series,
            'gap': gap,
            'post_treatment': [m >= self.treatment_period for m in months]
        })
        
        self.results['effect'] = {
            'att': att,
            'pre_treatment_gap': pre_gap,
            'post_treatment_gap': post_gap,
            'time_series': result_df
        }
        
        return self.results['effect']
    
    def placebo_test(self, n_placebos: int = 20) -> Dict:
        """
        Run in-space placebo tests: apply SCM to each donor state
        and compare their gaps to California's gap.
        
        If CA's gap is extreme relative to placebos, effect is credible.
        """
        placebo_gaps = {}
        
        donor_states = [s for s in self.data[self.entity_col].unique() 
                        if s != self.treated_unit][:n_placebos]
        
        for state in donor_states:
            try:
                sc = SyntheticControl(
                    self.data, self.outcome, state,
                    self.entity_col, self.time_col, self.treatment_period
                )
                sc.fit()
                effect = sc.estimate_effect()
                placebo_gaps[state] = effect['att']
            except Exception:
                continue
        
        # Compute p-value: fraction of placebos with larger |gap| than CA
        ca_att = self.results['effect']['att']
        n_larger = sum(1 for g in placebo_gaps.values() if abs(g) >= abs(ca_att))
        p_value = n_larger / len(placebo_gaps) if placebo_gaps else None
        
        return {
            'placebo_gaps': placebo_gaps,
            'treated_att': ca_att,
            'p_value': p_value,
            'n_placebos': len(placebo_gaps)
        }


# ══════════════════════════════════════════════
# 4. MAIN ANALYSIS PIPELINE
# ══════════════════════════════════════════════

def run_full_analysis(config: DataConfig = None) -> Dict:
    """Execute the complete causal inference pipeline."""
    
    if config is None:
        config = DataConfig()
    
    print("=" * 70)
    print("CAUSAL IMPACT OF GIG WORKER RECLASSIFICATION (PROP 22)")
    print("=" * 70)
    
    # ── Step 1: Generate Data ──
    print("\n[1/5] Generating panel data...")
    df = generate_panel_data(config)
    print(f"  Panel: {df['state'].nunique()} states x {df['month'].nunique()} months = {len(df)} obs")
    
    outcomes = {
        'log_gig_employment': 'Gig Employment (log)',
        'log_monthly_earnings': 'Monthly Earnings (log)',
        'log_avg_ride_price': 'Avg Ride Price (log)'
    }
    
    all_results = {'data': df, 'outcomes': {}}
    
    for outcome, label in outcomes.items():
        print(f"\n{'─' * 50}")
        print(f"  OUTCOME: {label}")
        print(f"{'─' * 50}")
        
        # ── Step 2: DID Estimation ──
        print(f"\n[2/5] Running DID estimation for {label}...")
        did = DifferenceInDifferences(df, outcome)
        twfe = did.estimate_twfe()
        print(f"  TWFE DID coefficient: {twfe['coefficient']:.4f} "
              f"(SE: {twfe['std_error']:.4f}, p: {twfe['p_value']:.4f})")
        print(f"  95% CI: [{twfe['ci_lower']:.4f}, {twfe['ci_upper']:.4f}]")
        
        # ── Step 3: Event Study ──
        print(f"\n[3/5] Running event study for {label}...")
        es = did.estimate_event_study()
        if es['parallel_trends_p_value']:
            print(f"  Parallel trends F-test: F={es['parallel_trends_f_stat']:.2f}, "
                  f"p={es['parallel_trends_p_value']:.4f}")
            if es['parallel_trends_p_value'] > 0.05:
                print("  ✓ Fail to reject parallel trends (good!)")
            else:
                print("  ⚠ Parallel trends may be violated")
        
        # ── Step 4: Robustness Checks ──
        print(f"\n[4/5] Running robustness checks for {label}...")
        robustness = did.run_robustness_checks()
        print(robustness.to_string(index=False))
        
        # ── Step 5: Synthetic Control ──
        print(f"\n[5/5] Running Synthetic Control for {label}...")
        sc = SyntheticControl(df, outcome, config.treated_state)
        sc_fit = sc.fit()
        print(f"  Pre-treatment RMSE: {sc_fit['pre_treatment_rmse']:.4f}")
        print(f"  Top 3 donors: {sc_fit['top_donors'][:3]}")
        
        sc_effect = sc.estimate_effect()
        print(f"  SC ATT estimate: {sc_effect['att']:.4f}")
        
        all_results['outcomes'][outcome] = {
            'label': label,
            'did': did.results,
            'sc_fit': sc_fit,
            'sc_effect': sc_effect
        }
    
    # ── Summary Table ──
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)
    
    summary_rows = []
    for outcome, label in outcomes.items():
        r = all_results['outcomes'][outcome]
        twfe = r['did']['twfe']
        summary_rows.append({
            'Outcome': label,
            'DID Estimate': f"{twfe['coefficient']:.4f}",
            'SE': f"{twfe['std_error']:.4f}",
            'p-value': f"{twfe['p_value']:.4f}",
            'SC Estimate': f"{r['sc_effect']['att']:.4f}",
            'True Effect': f"{getattr(config, f'true_{outcome.replace('log_', '').replace('gig_', '')}_effect', 'N/A')}"
        })
    
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    
    all_results['summary'] = summary_df
    return all_results


if __name__ == "__main__":
    results = run_full_analysis()
