import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns
import pandas as pd
from scipy.stats import kruskal
from scipy.stats import f_oneway
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
        title = metric + ' (MB)' if metric == 'Memory Usage' else metric + ' (seconds)' if metric == 'Processing Time' else metric

        sns.boxplot(ax=axes[row], x='model', y=metric, hue='technique', data=filtered_data, palette='Set2', showfliers=False)
        axes[row].set_title(f'{title} by Technique and Model')
        axes[row].grid(True, axis='y', linestyle='--', alpha=0.7)
        axes[row].set_xlabel('Model')
        axes[row].set_ylabel(title)
        axes[row].legend(title='Technique')
        row += 1

    # Kernel Linear Density plots by Model
    for metric in metrics:
        filtered_data = dataset
        title = metric + ' (MB)' if metric == 'Memory Usage' else metric + ' (seconds)' if metric == 'Processing Time' else metric

        if metric in ['F1-Score', 'ROC AUC', 'Precision', 'Accuracy', 'Recall']:
            sns.kdeplot(data=filtered_data, x=metric, hue='model', multiple='layer', palette='Set1', ax=axes[row], fill=True, bw_adjust=1, clip=(0, 1), linewidth=2, alpha=0.5)
        else:
            sns.kdeplot(data=filtered_data, x=metric, hue='model', multiple='layer', palette='Set1', ax=axes[row], fill=True, bw_adjust=1, clip=(0, None), linewidth=2, alpha=0.5)

        axes[row].set_title(f'{title} by Model')
        axes[row].set_xlabel(title)
        axes[row].set_ylabel('Frequency')
        axes[row].grid(True, linestyle='--', alpha=0.7)
        groups = [filtered_data[filtered_data['model'] == model][metric] for model in filtered_data['model'].unique()]
        if len(groups) > 1 and groups[0].size > 3:
            stat_kruskal, p_value_kruskal = kruskal(*groups)
            stat_anova, p_value_anova = f_oneway(*groups)
            y_pos = axes[row].get_ylim()[0] - 0.2

            text_content = f'Kruskal-Wallis stat: {stat_kruskal:.2f}, p: {p_value_kruskal:.2f}\nANOVA stat: {stat_anova:.2f}, p: {p_value_anova:.2f}'

            axes[row].text(
                0.5, y_pos, text_content,
                fontsize=10, ha='center', va='top', transform=axes[row].transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
            )
        row += 1

    # Kernel Linear Density plots by Technique
    for metric in metrics:
        filtered_data = dataset
        title = metric + ' (MB)' if metric == 'Memory Usage' else metric + ' (seconds)' if metric == 'Processing Time' else metric

        if metric in ['F1-Score', 'ROC AUC', 'Precision', 'Accuracy', 'Recall']:
            sns.kdeplot(data=filtered_data, x=metric, hue='technique', multiple='layer', palette='Set2', ax=axes[row], fill=True, bw_adjust=1, clip=(0, 1), linewidth=2, alpha=0.5)
        else:
            sns.kdeplot(data=filtered_data, x=metric, hue='technique', multiple='layer', palette='Set2', ax=axes[row], fill=True, bw_adjust=1, clip=(0, None), linewidth=2, alpha=0.5)

        axes[row].set_title(f'{title} by Technique')
        axes[row].set_xlabel(title)
        axes[row].set_ylabel('Frequency')
        axes[row].grid(True, linestyle='--', alpha=0.7)
        groups = [filtered_data[filtered_data['technique'] == technique][metric] for technique in filtered_data['technique'].unique()]
        if len(groups) > 1 and groups[0].size > 3:
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

    if num_metrics == 6:
        fig.subplots_adjust(top=1, bottom=0.0, hspace=0.6)
    if num_metrics == 3:
        fig.subplots_adjust(top=0.99, bottom=0.01, hspace=0.6)
    elif num_metrics == 2:
        fig.subplots_adjust(top=0.96, bottom=0.04, hspace=0.6)
    elif num_metrics == 1:
        fig.subplots_adjust(top=0.93, bottom=0.07, hspace=0.6)
    else:
        fig.subplots_adjust(top=0.98, bottom=0.02, hspace=0.6)

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