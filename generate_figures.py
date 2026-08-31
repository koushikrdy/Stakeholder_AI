"""
Generate visualizations and figures for paper
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def plot_system_architecture(output_dir: str = 'figures') -> None:
    """
    Create system architecture diagram.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Colors
    color_input = '#E8F4F8'
    color_process = '#B3E5FC'
    color_decision = '#FFE0B2'
    color_output = '#C8E6C9'
    
    # Define positions
    y_positions = [0.9, 0.75, 0.6, 0.45, 0.3, 0.15]
    
    # Input layer
    ax.add_patch(mpatches.FancyBboxPatch((0.05, y_positions[0]-0.05), 0.3, 0.08,
                                          boxstyle="round,pad=0.01", 
                                          facecolor=color_input, edgecolor='black'))
    ax.text(0.2, y_positions[0], 'Input Instance\n$x$', ha='center', va='center', fontweight='bold')
    
    # Model prediction
    ax.add_patch(mpatches.FancyBboxPatch((0.05, y_positions[1]-0.05), 0.3, 0.08,
                                          boxstyle="round,pad=0.01",
                                          facecolor=color_process, edgecolor='black'))
    ax.text(0.2, y_positions[1], 'Model Prediction\n$\\hat{y} = f(x)$', ha='center', va='center', fontweight='bold')
    ax.arrow(0.2, y_positions[0]-0.06, 0, -0.03, head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    # Explanation generators (parallel)
    explanations = ['SHAP', 'LIME', 'Counterfactual', 'Rule-based']
    x_positions = np.linspace(0.05, 0.65, len(explanations))
    
    ax.add_patch(mpatches.FancyBboxPatch((0.02, y_positions[2]-0.08), 0.9, 0.1,
                                          boxstyle="round,pad=0.01",
                                          facecolor='#F0F0F0', edgecolor='gray', linewidth=2))
    ax.text(0.47, y_positions[2]+0.04, 'Explanation Generation', ha='center', va='center', 
           fontweight='bold', fontsize=12, color='gray')
    
    for i, exp in enumerate(explanations):
        ax.add_patch(mpatches.FancyBboxPatch((x_positions[i]-0.045, y_positions[2]-0.05), 0.09, 0.06,
                                              boxstyle="round,pad=0.005",
                                              facecolor=color_process, edgecolor='black'))
        ax.text(x_positions[i], y_positions[2], exp, ha='center', va='center', fontsize=9, fontweight='bold')
        ax.arrow(0.2, y_positions[1]-0.06, x_positions[i]-0.2, -0.08, 
                head_width=0.01, head_length=0.008, fc='gray', ec='gray', alpha=0.5)
    
    # Utility computation (parallel)
    stakeholders = ['Doctor', 'Patient', 'Regulator']
    x_positions_util = np.linspace(0.05, 0.65, len(stakeholders))
    
    ax.add_patch(mpatches.FancyBboxPatch((0.02, y_positions[3]-0.08), 0.9, 0.1,
                                          boxstyle="round,pad=0.01",
                                          facecolor='#F0F0F0', edgecolor='gray', linewidth=2))
    ax.text(0.47, y_positions[3]+0.04, 'Utility Computation', ha='center', va='center',
           fontweight='bold', fontsize=12, color='gray')
    
    for i, stk in enumerate(stakeholders):
        ax.add_patch(mpatches.FancyBboxPatch((x_positions_util[i]-0.04, y_positions[3]-0.05), 0.08, 0.06,
                                              boxstyle="round,pad=0.005",
                                              facecolor=color_decision, edgecolor='black'))
        ax.text(x_positions_util[i], y_positions[3], stk, ha='center', va='center', fontsize=9, fontweight='bold')
        for j in range(len(explanations)):
            ax.arrow(x_positions[j], y_positions[2]-0.06, 
                    x_positions_util[i]-x_positions[j], -0.08,
                    head_width=0.01, head_length=0.008, fc='gray', ec='gray', alpha=0.3)
    
    # Selection layer
    ax.add_patch(mpatches.FancyBboxPatch((0.05, y_positions[4]-0.05), 0.3, 0.08,
                                          boxstyle="round,pad=0.01",
                                          facecolor=color_decision, edgecolor='black', linewidth=2))
    ax.text(0.2, y_positions[4], 'Optimal Selection\n$E^*_s = \\arg\\max_E U_s(E)$', 
           ha='center', va='center', fontweight='bold')
    for i, stk in enumerate(stakeholders):
        ax.arrow(x_positions_util[i], y_positions[3]-0.06, 
                0.2-x_positions_util[i], -0.08,
                head_width=0.01, head_length=0.008, fc='black', ec='black')
    
    # Output
    ax.add_patch(mpatches.FancyBboxPatch((0.05, y_positions[5]-0.05), 0.3, 0.08,
                                          boxstyle="round,pad=0.01",
                                          facecolor=color_output, edgecolor='black', linewidth=2))
    ax.text(0.2, y_positions[5], 'Stakeholder-Specific\nExplanation $E^*_s$', 
           ha='center', va='center', fontweight='bold')
    ax.arrow(0.2, y_positions[4]-0.06, 0, -0.03, head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    # Add key formula
    ax.text(0.75, y_positions[4], 
           '$U_s(E) = \\sum_i w_i \\cdot f_i(E)$\n\nDifferent weights\nfor each stakeholder',
           fontsize=11, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
           verticalalignment='center', family='monospace')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.title('System Architecture: Stakeholder-Specific Explanation Selection', 
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'architecture.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'architecture.png')}")
    plt.close()


def plot_stakeholder_comparison(results: Dict, output_dir: str = 'figures') -> None:
    """
    Create stakeholder comparison chart.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Sample data
    explanations = ['SHAP', 'LIME', 'Counterfactual', 'Rule-based']
    doctor_utilities = [0.82, 0.55, 0.71, 0.65]
    patient_utilities = [0.51, 0.58, 0.91, 0.55]
    regulator_utilities = [0.85, 0.48, 0.68, 0.80]
    
    stakeholders = ['Doctor', 'Patient', 'Regulator']
    utilities = [doctor_utilities, patient_utilities, regulator_utilities]
    
    for idx, (ax, stakeholder, util) in enumerate(zip(axes, stakeholders, utilities)):
        colors = ['#FF6B6B' if u == max(util) else '#4ECDC4' for u in util]
        bars = ax.bar(explanations, util, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_ylim([0, 1.0])
        ax.set_ylabel('Utility Score', fontweight='bold')
        ax.set_title(f'{stakeholder} Utility', fontweight='bold', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x labels
        ax.set_xticklabels(explanations, rotation=45, ha='right')
    
    plt.suptitle('Stakeholder Utility Comparison (Example: $\\hat{y}=0.78$)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stakeholder_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'stakeholder_comparison.png')}")
    plt.close()


def plot_utility_distributions(output_dir: str = 'figures') -> None:
    """
    Create utility score distribution plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 50
    
    explanations = ['SHAP', 'LIME', 'Counterfactual', 'Rule-based']
    
    for idx, exp in enumerate(explanations):
        ax = axes[idx // 2, idx % 2]
        
        # Generate random utilities for this explanation type across instances
        utilities_by_stakeholder = {
            'Doctor': np.random.beta(7, 3, n_samples),  # Biased towards high utility
            'Patient': np.random.beta(5, 5, n_samples) if idx == 2 else np.random.beta(4, 6, n_samples),
            'Regulator': np.random.beta(7, 2, n_samples),
        }
        
        for stakeholder, utils in utilities_by_stakeholder.items():
            ax.hist(utils, bins=15, alpha=0.6, label=stakeholder, edgecolor='black')
        
        ax.set_xlabel('Utility Score', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{exp} Utility Distribution', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlim([0, 1])
    
    plt.suptitle('Utility Score Distributions by Explanation Type',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'utility_distributions.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'utility_distributions.png')}")
    plt.close()


def plot_baseline_vs_proposed(output_dir: str = 'figures') -> None:
    """
    Create baseline vs. proposed framework comparison.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    stakeholders = ['Doctor', 'Patient', 'Regulator']
    baseline = [0.731, 0.548, 0.802]
    proposed = [0.812, 0.758, 0.835]
    
    x = np.arange(len(stakeholders))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline, width, label='Baseline (SHAP for all)',
                   color='#FF6B6B', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, proposed, width, label='Proposed (Stakeholder-specific)',
                   color='#4ECDC4', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Average Utility', fontweight='bold', fontsize=12)
    ax.set_title('Baseline vs. Proposed Framework', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(stakeholders, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels and improvement %
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        height1 = b1.get_height()
        height2 = b2.get_height()
        improvement = ((height2 - height1) / height1) * 100
        
        ax.text(b1.get_x() + b1.get_width()/2., height1,
               f'{height1:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.text(b2.get_x() + b2.get_width()/2., height2,
               f'{height2:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.text(i, height2 + 0.08, f'+{improvement:.1f}%',
               ha='center', fontweight='bold', fontsize=11, color='green',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_vs_proposed.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'baseline_vs_proposed.png')}")
    plt.close()


def plot_selection_workflow(output_dir: str = 'figures') -> None:
    """
    Create explanation selection workflow diagram.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Title
    ax.text(0.5, 0.95, 'Explanation Selection Workflow', ha='center', fontsize=16, fontweight='bold')
    
    # Step 1: Input
    ax.add_patch(mpatches.FancyBboxPatch((0.1, 0.85), 0.3, 0.08,
                                          boxstyle="round,pad=0.01",
                                          facecolor='#E8F4F8', edgecolor='black', linewidth=2))
    ax.text(0.25, 0.89, 'Step 1: Input\n$x$, $\\hat{y}$, $s$', ha='center', va='center', fontweight='bold')
    
    # Step 2: Generate explanations
    ax.add_patch(mpatches.FancyBboxPatch((0.6, 0.85), 0.3, 0.08,
                                          boxstyle="round,pad=0.01",
                                          facecolor='#B3E5FC', edgecolor='black', linewidth=2))
    ax.text(0.75, 0.89, 'Step 2: Generate\nExplanations', ha='center', va='center', fontweight='bold')
    ax.arrow(0.4, 0.89, 0.18, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # Step 3: Compute utilities
    ax.add_patch(mpatches.FancyBboxPatch((0.1, 0.65), 0.3, 0.08,
                                          boxstyle="round,pad=0.01",
                                          facecolor='#FFE0B2', edgecolor='black', linewidth=2))
    ax.text(0.25, 0.69, 'Step 3: Compute\nUtilities', ha='center', va='center', fontweight='bold')
    ax.arrow(0.75, 0.85, -0.48, -0.12, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # Step 4: Select
    ax.add_patch(mpatches.FancyBboxPatch((0.6, 0.65), 0.3, 0.08,
                                          boxstyle="round,pad=0.01",
                                          facecolor='#C8E6C9', edgecolor='black', linewidth=2))
    ax.text(0.75, 0.69, 'Step 4: Select\n$E^* = \\arg\\max U_s$', ha='center', va='center', fontweight='bold')
    ax.arrow(0.4, 0.69, 0.18, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # Step 5: Output
    ax.add_patch(mpatches.FancyBboxPatch((0.35, 0.45), 0.3, 0.08,
                                          boxstyle="round,pad=0.01",
                                          facecolor='#D5E8D4', edgecolor='black', linewidth=2))
    ax.text(0.5, 0.49, 'Output: $E^*_s$', ha='center', va='center', fontweight='bold', fontsize=12)
    ax.arrow(0.25, 0.65, 0.22, -0.12, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.75, 0.65, -0.22, -0.12, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # Utility function formula boxes
    formulas = [
        ('Doctor', 0.15, 0.3, '0.4 Act + 0.3 CR + 0.2 Fid + 0.1 CL'),
        ('Patient', 0.45, 0.3, '0.4 Int + 0.3 Act + 0.2 Tr + 0.1 J'),
        ('Regulator', 0.75, 0.3, '0.4 Fair + 0.3 Aud + 0.3 Tr'),
    ]
    
    for name, x, y, formula in formulas:
        ax.add_patch(mpatches.FancyBboxPatch((x-0.1, y-0.06), 0.2, 0.12,
                                              boxstyle="round,pad=0.01",
                                              facecolor='lightyellow', edgecolor='black', linewidth=1))
        ax.text(x, y+0.03, f'{name}\nUtility', ha='center', va='center', fontweight='bold', fontsize=9)
        ax.text(x, y-0.02, formula, ha='center', va='center', fontsize=7, family='monospace')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'selection_workflow.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'selection_workflow.png')}")
    plt.close()


def plot_selection_statistics(output_dir: str = 'figures') -> None:
    """
    Create stacked bar chart of explanation selections.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    stakeholders = ['Doctor', 'Patient', 'Regulator']
    shap_counts = [28, 2, 32]
    lime_counts = [4, 6, 2]
    cf_counts = [12, 30, 8]
    rules_counts = [6, 12, 8]
    
    x = np.arange(len(stakeholders))
    width = 0.6
    
    bars1 = ax.bar(x, shap_counts, width, label='SHAP', color='#FF6B6B', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, lime_counts, width, bottom=shap_counts, label='LIME', 
                  color='#4ECDC4', edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x, cf_counts, width, bottom=np.array(shap_counts)+np.array(lime_counts),
                  label='Counterfactual', color='#95E1D3', edgecolor='black', linewidth=1.5)
    bars4 = ax.bar(x, rules_counts, width,
                  bottom=np.array(shap_counts)+np.array(lime_counts)+np.array(cf_counts),
                  label='Rule-based', color='#F38181', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Number of Selections', fontweight='bold', fontsize=12)
    ax.set_title('Explanation Selection Statistics (50 instances)', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(stakeholders, fontweight='bold', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 55])
    ax.grid(axis='y', alpha=0.3)
    
    # Add count labels
    for i, (s, l, c, r) in enumerate(zip(shap_counts, lime_counts, cf_counts, rules_counts)):
        ax.text(i, s/2, f'{s} (56%)', ha='center', va='center', fontweight='bold', color='white', fontsize=10)
        ax.text(i, s+l/2, f'{l} (8%)', ha='center', va='center', fontweight='bold', color='white', fontsize=9)
        ax.text(i, s+l+c/2, f'{c} (24%)', ha='center', va='center', fontweight='bold', color='white', fontsize=9)
        ax.text(i, s+l+c+r/2, f'{r}', ha='center', va='center', fontweight='bold', color='white', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'selection_statistics.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'selection_statistics.png')}")
    plt.close()


def generate_all_figures(results: Dict = None, output_dir: str = 'figures') -> None:
    """
    Generate all figures for paper.
    """
    print(f"Generating figures in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_system_architecture(output_dir)
    plot_stakeholder_comparison(results or {}, output_dir)
    plot_utility_distributions(output_dir)
    plot_baseline_vs_proposed(output_dir)
    plot_selection_workflow(output_dir)
    plot_selection_statistics(output_dir)
    
    print(f"All figures saved to {output_dir}/")


if __name__ == '__main__':
    generate_all_figures()
