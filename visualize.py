"""
Visualization module for Causal Impact analysis.
Generates publication-quality figures for the DID and Synthetic Control results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from typing import Dict

# Set publication style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def plot_event_study(es_df: pd.DataFrame, outcome_label: str,
                     treatment_period: int = 0, save_path: str = None) -> plt.Figure:
    """
    Plot event study coefficients with confidence intervals.
    Publication-quality version following best practices.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Pre-treatment coefficients in blue, post in red
    pre = es_df[es_df['relative_time'] < 0]
    post = es_df[es_df['relative_time'] >= 0]
    
    # Confidence intervals
    ax.fill_between(es_df['relative_time'], es_df['ci_lower'], es_df['ci_upper'],
                    alpha=0.15, color='#3B82F6')
    
    # Point estimates
    ax.plot(pre['relative_time'], pre['coefficient'], 'o-', color='#3B82F6',
            markersize=6, linewidth=1.5, label='Pre-treatment')
    ax.plot(post['relative_time'], post['coefficient'], 's-', color='#EF4444',
            markersize=6, linewidth=1.5, label='Post-treatment')
    
    # Reference lines
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.axvline(x=-0.5, color='gray', linewidth=1.2, linestyle='--', alpha=0.7,
               label='Treatment onset')
    
    ax.set_xlabel('Months Relative to Prop 22 Implementation')
    ax.set_ylabel(f'Estimated Effect on {outcome_label}')
    ax.set_title(f'Event Study: Effect of Prop 22 on {outcome_label}')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    sns.despine()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_synthetic_control(sc_effect: Dict, outcome_label: str,
                           treatment_period: int = 12,
                           save_path: str = None) -> plt.Figure:
    """Plot treated unit vs synthetic control with gap."""
    
    ts = sc_effect['time_series']
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1],
                              gridspec_kw={'hspace': 0.3})
    
    # Top panel: levels
    ax1 = axes[0]
    ax1.plot(ts['month'], ts['treated'], 'o-', color='#EF4444',
             markersize=4, linewidth=2, label='California (Treated)')
    ax1.plot(ts['month'], ts['synthetic'], 's--', color='#3B82F6',
             markersize=4, linewidth=2, label='Synthetic California')
    ax1.axvline(x=treatment_period - 0.5, color='gray', linestyle='--',
                linewidth=1.2, alpha=0.7)
    ax1.fill_betweenx(ax1.get_ylim(), treatment_period - 0.5, ts['month'].max(),
                       alpha=0.05, color='red')
    ax1.set_ylabel(outcome_label)
    ax1.set_title(f'Synthetic Control: {outcome_label}')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: gap
    ax2 = axes[1]
    pre_gap = ts[ts['month'] < treatment_period]['gap']
    post_gap = ts[ts['month'] >= treatment_period]['gap']
    
    ax2.bar(ts[ts['month'] < treatment_period]['month'], pre_gap,
            color='#94A3B8', alpha=0.7, label='Pre-treatment gap')
    ax2.bar(ts[ts['month'] >= treatment_period]['month'], post_gap,
            color='#EF4444', alpha=0.7, label='Post-treatment gap')
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.axvline(x=treatment_period - 0.5, color='gray', linestyle='--', linewidth=1.2)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Gap (Treated - Synthetic)')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    sns.despine()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_parallel_trends(data: pd.DataFrame, outcome: str, outcome_label: str,
                         treatment_period: int = 12, save_path: str = None) -> plt.Figure:
    """Plot pre-treatment trends for treated vs control group."""
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Average outcome by treatment status and month
    trends = data.groupby(['month', 'treated'])[outcome].mean().reset_index()
    
    treated = trends[trends['treated'] == 1]
    control = trends[trends['treated'] == 0]
    
    ax.plot(treated['month'], treated[outcome], 'o-', color='#EF4444',
            linewidth=2, markersize=5, label='California')
    ax.plot(control['month'], control[outcome], 's-', color='#3B82F6',
            linewidth=2, markersize=5, label='Control States (avg)')
    
    ax.axvline(x=treatment_period - 0.5, color='gray', linestyle='--',
               linewidth=1.2, label='Prop 22 Implementation')
    
    ax.set_xlabel('Month')
    ax.set_ylabel(outcome_label)
    ax.set_title(f'Parallel Trends Check: {outcome_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    sns.despine()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_robustness_table(robustness_df: pd.DataFrame, outcome_label: str,
                          save_path: str = None) -> plt.Figure:
    """Create a coefficient plot for robustness specifications."""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    y_pos = range(len(robustness_df))
    coefs = robustness_df['coefficient'].values
    ses = robustness_df['std_error'].values
    
    colors = ['#EF4444' if p < 0.05 else '#94A3B8' 
              for p in robustness_df['p_value'].values]
    
    ax.barh(y_pos, coefs, xerr=1.96 * ses, color=colors, alpha=0.7,
            edgecolor='white', linewidth=0.5, capsize=3)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(robustness_df['specification'].values)
    ax.set_xlabel('DID Coefficient')
    ax.set_title(f'Robustness Checks: {outcome_label}')
    ax.grid(True, alpha=0.3, axis='x')
    sns.despine()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def generate_all_figures(results: Dict, output_dir: str = 'figures'):
    """Generate all figures from the analysis results."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for outcome, res in results['outcomes'].items():
        label = res['label']
        safe_name = outcome.replace('log_', '')
        
        # Event study
        if 'event_study' in res['did']:
            plot_event_study(
                res['did']['event_study']['event_study_df'], label,
                save_path=f'{output_dir}/event_study_{safe_name}.png'
            )
        
        # Synthetic control
        if 'effect' in res.get('sc_effect', {}):
            plot_synthetic_control(
                res['sc_effect'], label,
                save_path=f'{output_dir}/synthetic_control_{safe_name}.png'
            )
        
        # Parallel trends
        plot_parallel_trends(
            results['data'], outcome, label,
            save_path=f'{output_dir}/parallel_trends_{safe_name}.png'
        )
        
        # Robustness
        if 'robustness' in res['did']:
            plot_robustness_table(
                res['did']['robustness'], label,
                save_path=f'{output_dir}/robustness_{safe_name}.png'
            )
    
    print(f"All figures saved to {output_dir}/")


if __name__ == "__main__":
    from analysis import run_full_analysis
    results = run_full_analysis()
    generate_all_figures(results)
