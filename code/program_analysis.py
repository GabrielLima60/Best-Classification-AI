'''
    This code prepares the data and performs the analysis of the given dataset, technique and model.
'''

import warnings
import time
import sys
import psutil
import time
import os
import threading
import tracemalloc
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB


from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import SMOTE


warnings.filterwarnings("ignore")

class Analysis:
    '''
        This class call the other two classes, Prepare_data and Perform_analysis
    '''
    def __init__(self, dataframe, technique, model='SVM'):
        prepared = PrepareData(dataframe)
        performed = PerformAnalysis(technique, model, prepared.x_train, \
                                                       prepared.x_test, \
                                                       prepared.y_train, \
                                                       prepared.y_test)
        self.f1_score = performed.f1_score

class PrepareData:
    '''
        This class receaves the dataframe and returns the x_train, x_test, y_train and y_test
    '''
    def __init__(self, dataframe):
        if 'id' in dataframe.columns:
            dataframe.drop('id', axis=1, inplace=True)
        dataframe = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)

        for column in dataframe.columns:
            dataframe[column] = dataframe[column].apply(lambda x: pd.to_numeric(x, errors='coerce')\
                                                        if isinstance(x, str) and any(char.isdigit() for char in x) else x)

        dataframe = dataframe.dropna()

        self.x = dataframe.iloc[:, :-1]
        self.y = dataframe.iloc[:, -1]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.20, random_state=42)

class PerformAnalysis:
    '''
        This class performs the analysis and saves the f1_score
    '''
    def __init__(self, technique, model, x_train, x_test, y_train, y_test):

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_test

        if technique == 'PCA':
            self.apply_pca()
        elif technique == 'IncPCA':
            self.apply_ipca()
        elif technique == 'ICA':
            self.apply_ica()
        elif technique == 'LDA':
            self.apply_lda()
        elif technique == 'SMOTE':
            self.apply_smote()
        else: # Apply no technique
            pass

        self.apply_normalization()

        self.f1_score = self.select_model_and_get_f1(model)

    def apply_pca(self):
        pca = PCA(n_components=0.95)
        self.x_train = pca.fit_transform(self.x_train)
        self.x_test = pca.transform(self.x_test)

    def apply_ipca(self):
        pca = PCA()
        pca.fit(self.x_train)
        n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
        ipca = IncrementalPCA(n_components=n_components)
        self.x_train = ipca.fit_transform(self.x_train)
        self.x_test = ipca.transform(self.x_test)

    def apply_ica(self):
        n_components = int(self.x_train.shape[1]/2)
        ica = FastICA(n_components=n_components, random_state=42)
        self.x_train = ica.fit_transform(self.x_train)
        self.x_test = ica.transform(self.x_test)

    def apply_lda(self):
        lda = LinearDiscriminantAnalysis()
        self.x_train = lda.fit_transform(self.x_train, self.y_train)

        self.x_test = lda.transform(self.x_test)

    def apply_smote(self):
        smote = SMOTE(random_state=42)
        self.x_train, self.y_train = smote.fit_resample(self.x_train, self.y_train)


    def apply_normalization(self):
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)

    def select_model_and_get_f1(self, model):
        model_dict = {'Naive Bayes': GaussianNB(),
                      'SVM': SVC(random_state = 42),
                      'MLP': MLPClassifier(random_state = 42),
                      'Tree': DecisionTreeClassifier(random_state = 42),
                      'KNN': KNeighborsClassifier(),
                      'LogReg': LogisticRegression(random_state = 42),
                      'GBC': GradientBoostingClassifier(random_state = 42)
                     }


        return self.get_f1_score(model_dict[model])

    def get_f1_score(self, classifier):
        classifier.fit(self.x_train, self.y_train)
        self.y_pred = classifier.predict(self.x_test)
        return f1_score(self.y_test, self.y_pred, average='weighted')

class MemoryMonitor:
    def __init__(self, pid, interval=1):
        self.process = psutil.Process(pid)
        self.interval = interval
        self.max_memory_rss = 0  

    def __call__(self):
        while True:
            memory_info = self.process.memory_info()
            rss_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            
            if rss_mb > self.max_memory_rss:
                self.max_memory_rss = rss_mb
                
            time.sleep(self.interval)


# MAIN

given_dataset = sys.argv[1]
given_dataset = pd.read_csv(given_dataset)
given_technique = sys.argv[2]
given_model = sys.argv[3]

# Getting the memory usage
current_pid = os.getpid()
memory_monitor = MemoryMonitor(current_pid)

monitor_thread = threading.Thread(target=memory_monitor)
monitor_thread.daemon = True
monitor_thread.start()

# We use tracemalloc snapshots to get the memory used immediately before the execution and after it.
tracemalloc.start()
start_time = time.time()
start_snapshot = tracemalloc.take_snapshot()


a = Analysis(given_dataset, given_technique, given_model)


end_snapshot = tracemalloc.take_snapshot()
end_time = time.time()
tracemalloc.stop()

# The memory usage is shown in Kibibytes (KiB)
diff_snapshot = end_snapshot.compare_to(start_snapshot, 'lineno')
memory_usage = sum(stat.size for stat in diff_snapshot)/1024

processing_time = end_time - start_time

result = str(given_technique) + "," + \
         str(given_model) + "," + \
         str(a.f1_score) + "," + \
         str(processing_time) + "," + \
         str(memory_monitor.max_memory_rss)

print(result)