import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns
import pandas as pd
from PIL import Image

def plot_all():
    dataset = pd.read_csv('results table//results.csv')
    metrics = dataset.columns.tolist()
    metrics.remove('technique')
    metrics.remove('model')
    num_metrics = len(metrics)

    # Calculate the number of rows based on metrics count, and set up the figure
    total_rows = num_metrics * 3  # 1 row each for box plot, model histogram, and technique histogram
    fig, axes = plt.subplots(total_rows, 1, figsize=(12, 5 * total_rows))
    
    row = 0

    # Section 1: Box Plots by Technique and Model
    for i, metric in enumerate(metrics):
        filtered_data = remove_outliers(dataset, metric)
        
        sns.boxplot(ax=axes[row], x='model', y=metric, hue='technique', data=filtered_data, palette='Set2', showfliers=False)
        axes[row].set_title(f'{metric} by Technique and Model (Without Outliers)')
        axes[row].grid(True, axis='y', linestyle='--', alpha=0.7)
        axes[row].set_xlabel('Model')
        axes[row].set_ylabel(metric + ' (MB)' if metric == 'Memory Usage' else metric + ' (seconds)' if metric == 'Processing Time' else metric)
        axes[row].legend(title='Technique')
        row += 1

    # Section 2: Histograms by Model
    for metric in metrics:
        filtered_data = remove_outliers(dataset, metric)
        sns.histplot(data=filtered_data, x=metric, hue='model', multiple='stack', palette='Set1', kde=True, ax=axes[row])
        axes[row].set_title(f'Histogram of {metric} by Model (Without Outliers)')
        axes[row].set_xlabel(metric)
        axes[row].set_ylabel('Frequency')
        axes[row].grid(True, linestyle='--', alpha=0.7)
        row += 1

    # Section 3: Histograms by Technique
    for metric in metrics:
        filtered_data = remove_outliers(dataset, metric)
        sns.histplot(data=filtered_data, x=metric, hue='technique', multiple='stack', palette='Set3', kde=True, ax=axes[row])
        axes[row].set_title(f'Histogram of {metric} by Technique (Without Outliers)')
        axes[row].set_xlabel(metric)
        axes[row].set_ylabel('Frequency')
        axes[row].grid(True, linestyle='--', alpha=0.7)
        row += 1

    # Adjust layout and add a section title for each plot type
    plt.tight_layout()
    fig.suptitle('Combined Analysis: Box Plots and Histograms', fontsize=18, fontweight='bold')
    fig.subplots_adjust(top=0.95)

    # Save the figure as one image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    image_data = canvas.tostring_rgb()
    size = canvas.get_width_height()
    image = Image.frombytes("RGB", size, image_data)
    image.save('results image\\combined_graphs.png')

def remove_outliers(df, metric):
    mean = df[metric].mean()
    std = df[metric].std()
    filtered_df = df[(df[metric] >= mean - 3 * std) & (df[metric] <= mean + 3 * std)]
    return filtered_df

if __name__ == "__main__":
    plot_all()
    sys.exit()
