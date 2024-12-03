import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns
import pandas as pd
from scipy.stats import kruskal
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_dunn
from PIL import Image

def plot_all():
    dataset = pd.read_csv('results table//results.csv')
    metrics = dataset.columns.tolist()
    metrics.remove('technique')
    metrics.remove('model')
    num_metrics = len(metrics)

    total_rows = num_metrics * 3  
    fig, axes = plt.subplots(total_rows, 1, figsize=(12, 5 * total_rows))
    axes = axes.flatten()
    
    row = 0
    
    # Box Plots by Technique and Model
    for i, metric in enumerate(metrics):
        filtered_data = dataset
        
        sns.boxplot(ax=axes[row], x='model', y=metric, hue='technique', data=filtered_data, palette='Set2', showfliers=False)
        axes[row].set_title(f'{metric} by Technique and Model')
        axes[row].grid(True, axis='y', linestyle='--', alpha=0.7)
        axes[row].set_xlabel('Model')
        axes[row].set_ylabel(metric + ' (MB)' if metric == 'Memory Usage' else metric + ' (seconds)' if metric == 'Processing Time' else metric)
        axes[row].legend(title='Technique')
        row += 1

    # Histograms by Model
    for metric in metrics:
        filtered_data = dataset

        sns.kdeplot(data=filtered_data, x=metric, hue='model', multiple='layer', palette='Set1', ax=axes[row], fill=False, bw_adjust=0.5)
        axes[row].set_title(f'Density Plot of {metric} by Model')
        axes[row].set_xlabel(metric)
        axes[row].set_ylabel('Frequency')
        axes[row].grid(True, linestyle='--', alpha=0.7)
        groups = [filtered_data[filtered_data['model'] == model][metric] for model in filtered_data['model'].unique()]
        if len(groups) > 1:
            stat_kruskal, p_value_kruskal = kruskal(*groups)
            stat_anova, p_value_anova = f_oneway(*groups)
            y_pos = axes[row].get_ylim()[0] - 0.2

            text_content = f'Kruskal-Wallis stat: {stat_kruskal:.2f}, p: {p_value_kruskal:.2f}\nANOVA stat: {stat_anova:.2f}, p: {p_value_anova:.2f}'

            '''
            # Post-Hoc Analysis for Kruskal-Wallis 
            if p_value_kruskal < 0.05:
                posthoc_dunn_results = posthoc_dunn(
                    filtered_data, val_col=metric, group_col='technique', p_adjust='bonferroni'
                )
                significant_pairs = [
                    f'{index[0]} vs {index[1]}: p={value:.2f}'
                    for index, value in posthoc_dunn_results.stack().items()
                    if index[0] != index[1] and value < 0.05
                ]
                text_content += '\nDunn\'s Post-Hoc:\n' + '\n'.join(significant_pairs)

            # Post-Hoc Analysis for ANOVA 
            if p_value_anova < 0.05:
                tukey_results = pairwise_tukeyhsd(
                    filtered_data[metric], filtered_data['technique'], alpha=0.05
                )
                tukey_summary = tukey_results.summary().data[1:]  # Skip header
                significant_pairs = [
                    f'{row[0]} vs {row[1]}: p={row[4]:.2f}'
                    for row in tukey_summary if row[4] < 0.05
                ]
                text_content += '\nTukey\'s Post-Hoc:\n' + '\n'.join(significant_pairs)
            '''

            # Add all test results to the graph
            axes[row].text(
                0.5, y_pos, text_content,
                fontsize=10, ha='center', va='top', transform=axes[row].transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
            )
        row += 1

    # Histograms by Technique
    for metric in metrics:
        filtered_data = dataset

        sns.kdeplot(data=filtered_data, x=metric, hue='technique', multiple='layer', palette='Set2', ax=axes[row], fill=False, bw_adjust=0.5)
        axes[row].set_title(f'Density Plot of {metric} by Technique')
        axes[row].set_xlabel(metric)
        axes[row].set_ylabel('Frequency')
        axes[row].grid(True, linestyle='--', alpha=0.7)
        groups = [filtered_data[filtered_data['technique'] == technique][metric] for technique in filtered_data['technique'].unique()]
        if len(groups) > 1:
            stat_kruskal, p_value_kruskal = kruskal(*groups)
            stat_anova, p_value_anova = f_oneway(*groups)
            y_pos = axes[row].get_ylim()[0] - 0.2
            axes[row].text(
                0.5, y_pos, f'Kruskal-Wallis   stat: {stat_kruskal:.2f}    p_value: {p_value_kruskal:.2f} \nANOVA   stat: {stat_anova:.2f}   p_value: {p_value_anova:.2f}',
                fontsize=10, ha='center', va='top',
                transform=axes[row].transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
            )
        row += 1

    plt.tight_layout()
    fig.suptitle('Results', fontsize=18, fontweight='bold')
    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.6)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    image_data = canvas.tostring_rgb()
    size = canvas.get_width_height()
    image = Image.frombytes("RGB", size, image_data)
    image.save('results image\\graphs.png')

def remove_outliers(df, metric):
    mean = df[metric].mean()
    std = df[metric].std()
    filtered_df = df[(df[metric] >= mean - 3 * std) & (df[metric] <= mean + 3 * std)]
    return filtered_df

if __name__ == "__main__":
    plot_all()
    sys.exit()