import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns
import pandas as pd
from PIL import Image

def plot():
    dataset = pd.read_csv('results table\\results.csv')

    metrics = ['F1-Score', 'Processing Time', 'ROC AUC', 'Memory Usage', 'Precision', 'Recall']

    for metric in metrics:
      filtered_data = remove_outliers(dataset, metric)
      
      plt.figure(figsize=(10, 6))
      
      # Create a dot plot
      sns.stripplot(x='model', y=metric, hue='technique', data=filtered_data, jitter=True, size=7, palette='Set2')
      
      # Add plot titles and labels
      plt.title(f'Dot Plot for {metric} by Technique and Model (Without Outliers)')
      plt.xlabel('Model')
      
      if metric == 'Memory Usage':
        plt.ylabel(metric + ( ' (MB)'))
      elif metric == 'Processing Time':
        plt.ylabel(metric + ' (seconds)')
      else:
        plt.ylabel(metric)

      plt.legend(title='Technique')

    plt.tight_layout()

    figure = plt.gcf()  
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    image_data = canvas.tostring_rgb()
    size = canvas.get_width_height()

    # Save the Seaborn plot as an image
    image = Image.frombytes("RGB", size, image_data)
    image.save('results image\\graphs.png')

def remove_outliers(df, metric):
    mean = df[metric].mean()
    std = df[metric].std()
    # Keep only the values within 3 standard deviations
    filtered_df = df[(df[metric] >= mean - 3 * std) & (df[metric] <= mean + 3 * std)]
    return filtered_df

if __name__ == "__main__":
    plot()
    sys.exit(0)
