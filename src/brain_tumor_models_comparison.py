"""
Brain Tumor Classification - Models Comparison
This code compares the performance of all models
Each cell is separated by: # ============== CELL X ==============
Run this AFTER training all individual models
"""

# ============== CELL 1: IMPORTS ==============
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

print("Models Comparison Tool")


# ============== CELL 2: MANUALLY ENTER YOUR RESULTS ==============
# After training all models, enter their test results here
# Replace these values with your actual results from each model

results = {
    'Model': ['Custom CNN', 'VGG16', 'VGG19', 'MobileNet', 'ResNet50'],
    'Accuracy': [0.00, 0.00, 0.00, 0.00, 0.00],  # <-- UPDATE THESE
    'Precision': [0.00, 0.00, 0.00, 0.00, 0.00],  # <-- UPDATE THESE
    'Recall': [0.00, 0.00, 0.00, 0.00, 0.00],  # <-- UPDATE THESE
    'F1 Score': [0.00, 0.00, 0.00, 0.00, 0.00],  # <-- UPDATE THESE
}

# Create DataFrame
df_results = pd.DataFrame(results)
print("\nModel Performance Comparison:")
print("="*80)
print(df_results.to_string(index=False))
print("="*80)


# ============== CELL 3: VISUALIZE MODELS COMPARISON ==============
# Set style
sns.set_style("whitegrid")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Brain Tumor Classification - Models Comparison', 
             fontsize=18, fontweight='bold', y=1.00)

# Color palette
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

# Plot 1: Accuracy Comparison
ax1 = axes[0, 0]
bars1 = ax1.bar(df_results['Model'], df_results['Accuracy'], color=colors, alpha=0.8)
ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_ylim([0, 1.0])
ax1.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Precision Comparison
ax2 = axes[0, 1]
bars2 = ax2.bar(df_results['Model'], df_results['Precision'], color=colors, alpha=0.8)
ax2.set_title('Precision Comparison', fontsize=14, fontweight='bold', pad=15)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_ylim([0, 1.0])
ax2.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: Recall Comparison
ax3 = axes[1, 0]
bars3 = ax3.bar(df_results['Model'], df_results['Recall'], color=colors, alpha=0.8)
ax3.set_title('Recall Comparison', fontsize=14, fontweight='bold', pad=15)
ax3.set_ylabel('Recall', fontsize=12)
ax3.set_ylim([0, 1.0])
ax3.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 4: F1 Score Comparison
ax4 = axes[1, 1]
bars4 = ax4.bar(df_results['Model'], df_results['F1 Score'], color=colors, alpha=0.8)
ax4.set_title('F1 Score Comparison', fontsize=14, fontweight='bold', pad=15)
ax4.set_ylabel('F1 Score', fontsize=12)
ax4.set_ylim([0, 1.0])
ax4.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()


# ============== CELL 4: ALL METRICS IN ONE PLOT ==============
# Grouped bar chart for all metrics
fig, ax = plt.subplots(figsize=(14, 8))

# Set the width of bars and positions
x = np.arange(len(df_results['Model']))
width = 0.2

# Create bars
bars1 = ax.bar(x - 1.5*width, df_results['Accuracy'], width, label='Accuracy', color='#FF6B6B', alpha=0.8)
bars2 = ax.bar(x - 0.5*width, df_results['Precision'], width, label='Precision', color='#4ECDC4', alpha=0.8)
bars3 = ax.bar(x + 0.5*width, df_results['Recall'], width, label='Recall', color='#45B7D1', alpha=0.8)
bars4 = ax.bar(x + 1.5*width, df_results['F1 Score'], width, label='F1 Score', color='#FFA07A', alpha=0.8)

# Add labels and title
ax.set_xlabel('Models', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Brain Tumor Classification - All Metrics Comparison', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(df_results['Model'], fontsize=11)
ax.legend(fontsize=12, loc='lower right')
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()


# ============== CELL 5: IDENTIFY BEST MODEL ==============
# Find best model for each metric
print("\n" + "="*80)
print("BEST MODELS FOR EACH METRIC")
print("="*80)

best_accuracy_idx = df_results['Accuracy'].idxmax()
best_precision_idx = df_results['Precision'].idxmax()
best_recall_idx = df_results['Recall'].idxmax()
best_f1_idx = df_results['F1 Score'].idxmax()

print(f"Best Accuracy:  {df_results.loc[best_accuracy_idx, 'Model']:12s} - {df_results.loc[best_accuracy_idx, 'Accuracy']:.4f}")
print(f"Best Precision: {df_results.loc[best_precision_idx, 'Model']:12s} - {df_results.loc[best_precision_idx, 'Precision']:.4f}")
print(f"Best Recall:    {df_results.loc[best_recall_idx, 'Model']:12s} - {df_results.loc[best_recall_idx, 'Recall']:.4f}")
print(f"Best F1 Score:  {df_results.loc[best_f1_idx, 'Model']:12s} - {df_results.loc[best_f1_idx, 'F1 Score']:.4f}")
print("="*80)

# Overall best model (based on F1 score)
overall_best_idx = df_results['F1 Score'].idxmax()
print(f"\n🏆 OVERALL BEST MODEL: {df_results.loc[overall_best_idx, 'Model']}")
print(f"   F1 Score: {df_results.loc[overall_best_idx, 'F1 Score']:.4f}")
print("="*80)

print("\n✅ Models Comparison - Complete!")
