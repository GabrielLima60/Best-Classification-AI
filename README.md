# Overview
Best-Classification-AI is a software able to read a CSV file given by the user and perform analysis on different supervisioned AI classification models and optimization techniques, comparing the resulting F1-Score, processing time and memory usage.
The results of the analysis are provided in the program though a series of graphs and be extracted at the results table and results image folders.

Run the .jar file to execute the software.

# installation 
Prior to running the program, ensure the following dependencies are installed on your device:
- Java Development Kit (JDK) and Java Runtime Environment (JRE)
- Python 3

Additionally, install the necessary Python libraries by executing the following command in your terminal:
```
pip install matplotlib seaborn pandas numpy scikit-learn pillow psutil
```

# ATENTION
- Your CSV must allow for supervisioned classification AIs.
- The target variable (y) column must be the last column of the CSV.
- This software does not perform proper data cleaning.

