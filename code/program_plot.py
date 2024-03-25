import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns
import pandas as pd
from PIL import Image

def plot():
    dataset = pd.read_csv('results table\\results.csv')

    metrics = ["f1_score", "processing_time", "memory_usage"]
    titles = ["F1 Score Comparison", "Processing Time Comparison (seconds)", "Memory Usage Comparison (KiB)"]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5 * len(metrics)), dpi=120)

    sns.set_palette("hls")

    for i, metric in enumerate(metrics):
        ax = sns.barplot(data=dataset, x="model", y=metric, hue="technique", errorbar=None, ax=axes[i])
        axes[i].set_xlabel("AI Model")
        axes[i].set_ylabel(metric)
        axes[i].set_title(titles[i])

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=2)

        for p in ax.patches:
            height = p.get_height() if not pd.isnull(p.get_height()) else 0
            if height != 0:  
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='center', fontsize=8, color='black', xytext=(0, 5),
                            textcoords='offset points')

    plt.tight_layout()

    figure = plt.gcf()  
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    image_data = canvas.tostring_rgb()
    size = canvas.get_width_height()

    # Save the Seaborn plot as an image
    image = Image.frombytes("RGB", size, image_data)
    image.save('results image\\graphs.png')

if __name__ == "__main__":
    plot()
    sys.exit(0)
