import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns
import pandas as pd
from PIL import Image

def plot():
    dataset = pd.read_csv('results table\\results.csv')

    metrics = ['F1-Score', 'Processing Time', 'ROC AUC', 'Memory Usage', 'Precision', 'Recall']
    num_metrics = len(metrics)
    
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 6 * num_metrics))  

    for i, metric in enumerate(metrics):
        filtered_data = remove_outliers(dataset, metric)
        
        sns.stripplot(ax=axes[i], x='model', y=metric, hue='technique', data=filtered_data, jitter=True, size=7, palette='Set2')
        
        axes[i].set_title(f'{metric} by Technique and Model (Without Outliers)')
        axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)
        axes[i].set_xlabel('Model')
        
        if metric == 'Memory Usage':
            axes[i].set_ylabel(metric + ' (MB)')
        elif metric == 'Processing Time':
            axes[i].set_ylabel(metric + ' (seconds)')
        else:
            axes[i].set_ylabel(metric)

        axes[i].legend(title='Technique')

    plt.tight_layout()

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
    plot()
    sys.exit(0)
