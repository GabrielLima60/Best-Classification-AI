import warnings
import time
import sys
import psutil
import time
import os
import threading
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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from scipy.stats import randint
from scipy.stats import uniform


warnings.filterwarnings("ignore")

class Analysis:
    '''
        This class call the other two classes, Prepare_data and Perform_analysis
    '''
    def __init__(self, dataframe, technique, optimization, cross_validation, model='SVM'):

        prepared = PrepareData(dataframe)

        if cross_validation == 'Hold-Out':
            performed = PerformAnalysis(technique, model, optimization, prepared.X_train, \
                                                                        prepared.X_test, \
                                                                        prepared.y_train, \
                                                                        prepared.y_test)
            self.f1_score = performed.f1_score

        elif cross_validation == 'K-Fold': 
            
            f1_score_list = []
            kf = KFold(n_splits=5)

            for train_index, test_index in kf.split(prepared.x):
                X_train, X_test = prepared.x.iloc[train_index], prepared.x.iloc[test_index]
                y_train, y_test = prepared.y.iloc[train_index], prepared.y.iloc[test_index]

                performed = PerformAnalysis(technique, model, optimization, X_train, \
                                                                            X_test, \
                                                                            y_train, \
                                                                            y_test)
                f1_score_list.append(performed.f1_score)
            self.f1_score = sum(f1_score_list) / len(f1_score_list)
                

        else:
            raise ValueError('Wrong cross-validation name given.')


class PrepareData:
    '''
        This class receaves the dataframe and returns the X_train, X_test, y_train and y_test
    '''
    def __init__(self, dataframe):
        if 'id' in dataframe.columns:
            dataframe.drop('id', axis=1, inplace=True)
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)

        for column in dataframe.columns:
            dataframe[column] = dataframe[column].apply(lambda x: pd.to_numeric(x, errors='coerce')\
                                                        if isinstance(x, str) and any(char.isdigit() for char in x) else x)

        dataframe = dataframe.dropna()

        self.x = dataframe.iloc[:, :-1]
        self.y = dataframe.iloc[:, -1]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.20)

class PerformAnalysis:
    '''
        This class performs the analysis and saves the f1_score
    '''
    def __init__(self, technique, model, optimization, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
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
        else: # Apply no technique
            pass

        self.apply_normalization()

        self.f1_score = self.select_model_and_get_f1(model, optimization)

    def apply_pca(self):
        pca = PCA(n_components=0.95)
        self.X_train = pca.fit_transform(self.X_train)
        self.X_test = pca.transform(self.X_test)

    def apply_ipca(self):
        pca = PCA()
        pca.fit(self.X_train)
        n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
        ipca = IncrementalPCA(n_components=n_components)
        self.X_train = ipca.fit_transform(self.X_train)
        self.X_test = ipca.transform(self.X_test)

    def apply_ica(self):
        n_components = int(self.X_train.shape[1]/2)
        ica = FastICA(n_components=n_components)
        self.X_train = ica.fit_transform(self.X_train)
        self.X_test = ica.transform(self.X_test)

    def apply_lda(self):
        lda = LinearDiscriminantAnalysis()
        self.X_train = lda.fit_transform(self.X_train, self.y_train)

        self.X_test = lda.transform(self.X_test)


    def apply_normalization(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def select_model_and_get_f1(self, model, optimization):
        model_dict = {'Naive Bayes': GaussianNB(),
                      'SVM': SVC(),
                      'MLP': MLPClassifier(),
                      'Tree': DecisionTreeClassifier(),
                      'KNN': KNeighborsClassifier(),
                      'LogReg': LogisticRegression(),
                      'GBC': GradientBoostingClassifier()
                     }
        
        param_grid_dict = {
            'Naive Bayes': {},
            'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
            'MLP': {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]},
            'Tree': {'max_depth': [None, 10, 20, 30]},
            'KNN': {'n_neighbors': [3, 5, 7]},
            'LogReg': {'C': [0.1, 1, 10]},
            'GBC': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
        }

        param_random_dict = {
            'Naive Bayes': {},
            'SVM': {'C': uniform(0.1, 10), 'kernel': ['linear', 'rbf']},
            'MLP': {'hidden_layer_sizes': [(50,), (100,)], 'alpha': uniform(0.0001, 0.001)},
            'Tree': {'max_depth': randint(10, 30)},
            'KNN': {'n_neighbors': randint(3, 10)},
            'LogReg': {'C': uniform(0.1, 10)},
            'GBC': {'n_estimators': randint(50, 200), 'learning_rate': uniform(0.01, 1)}
        }
        
        if optimization == 'Grid Search':
            optimized_model = GridSearchCV(estimator=model_dict[model], param_grid = param_grid_dict[model], cv=5)
        elif optimization == 'Random Search':
            optimized_model = RandomizedSearchCV(estimator = model_dict[model], param_distributions = param_random_dict[model], n_iter=50, cv=5)
        else:
            raise ValueError('Wrong optimization name given')


        return self.get_f1_score(optimized_model)

    def get_f1_score(self, classifier):
        classifier.fit(self.X_train, self.y_train)
        self.y_pred = classifier.predict(self.X_test)
        return f1_score(self.y_test, self.y_pred, average='weighted')

class MemoryMonitor:
    def __init__(self, pid, interval=0.001):
        self.process = psutil.Process(pid)
        self.interval = interval
        self.initial_memory_usage = None
        self.max_memory_rss = 0  

    def __call__(self):
        while True:
            memory_info = self.process.memory_info()
            rss_mb = memory_info.rss / (1024 * 1024)  # Convert to MB

            if self.initial_memory_usage is None:
                self.initial_memory_usage = rss_mb
            
            if rss_mb > self.max_memory_rss:
                self.max_memory_rss = rss_mb
                
            time.sleep(self.interval)


# MAIN

given_dataset = sys.argv[1]
given_dataset = pd.read_csv(given_dataset)
given_technique = sys.argv[2]
given_model = sys.argv[3]
given_optimization = sys.argv[4]
given_cross_validation = sys.argv[5]
given_parameters = sys.argv[6]

# Getting the memory usage
current_pid = os.getpid()
memory_monitor = MemoryMonitor(current_pid)

monitor_thread = threading.Thread(target=memory_monitor)
monitor_thread.daemon = True
monitor_thread.start()

start_time = time.time()

a = Analysis(given_dataset, given_technique, given_optimization, given_cross_validation, given_model)

end_time = time.time()

processing_time = end_time - start_time

result = str(given_technique) + "," + \
         str(given_model) + "," + \
         str(a.f1_score) + "," + \
         str(processing_time) + "," + \
         str(memory_monitor.max_memory_rss - memory_monitor.initial_memory_usage)

print(result)
