import warnings
import time
import sys
import psutil
import os
import threading
import pandas as pd
import numpy as np
import json
import ast
import importlib.util

import xgboost as xgb

from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.metrics import f1_score, roc_curve, auc, recall_score, precision_score, accuracy_score,roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.decomposition import PCA, IncrementalPCA, FastICA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from scipy.stats import randint, uniform

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
            self.precision = performed.precision
            self.accuracy = performed.accuracy
            self.recall = performed.recall
            self.roc_auc = performed.roc_auc

        elif cross_validation == 'K-Fold': 
            
            f1_score_list = []
            precision_list = []
            accuracy_list = []
            recall_list = []
            roc_auc_list = []

            kf = StratifiedKFold(n_splits=5, shuffle=True)

            for train_index, test_index in kf.split(prepared.x, prepared.y):
                X_train, X_test = prepared.x.iloc[train_index], prepared.x.iloc[test_index]
                y_train, y_test = prepared.y.iloc[train_index], prepared.y.iloc[test_index]

                performed = PerformAnalysis(technique, model, optimization, X_train, \
                                                                            X_test, \
                                                                            y_train, \
                                                                            y_test)
                f1_score_list.append(performed.f1_score)
                precision_list.append(performed.precision)
                accuracy_list.append(performed.accuracy)
                recall_list.append(performed.recall)
                roc_auc_list.append(performed.roc_auc)

            self.f1_score = sum(f1_score_list) / len(f1_score_list)
            self.precision = sum(precision_list) / len(precision_list)
            self.accuracy = sum(accuracy_list) / len(accuracy_list)
            self.recall = sum(recall_list) / len(recall_list)
            self.roc_auc = sum(roc_auc_list) / len(roc_auc_list)
                

        else:
            raise ValueError('Wrong cross-validation name given.')


class PrepareData:
    '''
        This class receaves the dataframe and returns the X_train, X_test, y_train and y_test
    '''
    def __init__(self, dataframe):
        self.x = dataframe.iloc[:, :-1]
        self.y = dataframe.iloc[:, -1]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, stratify=self.y,test_size=0.20, train_size=0.80)
    

class PerformAnalysis:
    '''
        This class performs the analysis and saves the f1_score
    '''
    def __init__(self, technique, model, optimization, X_train, X_test, y_train, y_test):

        self.X_train = X_train.to_numpy()
        self.X_test = X_test.to_numpy()
        self.y_train = y_train.to_numpy()
        self.y_test = y_test.to_numpy()
        self.y_pred = None

        if technique == 'PCA':
            self.apply_pca()
        elif technique == 'IncPCA':
            self.apply_ipca()
        elif technique == 'ICA':
            self.apply_ica()
        elif technique == 'LDA':
            self.apply_lda()
        elif technique == 'No Technique':
            pass
        else: 
            pass

        self.select_model(model, optimization)

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
        n_components = int(self.X_train.shape[1]*0.80)
        ica = FastICA(n_components=n_components)
        self.X_train = ica.fit_transform(self.X_train)
        self.X_test = ica.transform(self.X_test)

    def apply_lda(self):
        lda = LinearDiscriminantAnalysis()
        self.X_train = lda.fit_transform(self.X_train, self.y_train)

        self.X_test = lda.transform(self.X_test)

    def load_optimization_data(self, json_file_path):
            with open(json_file_path, 'r') as file:
                data = json.load(file)

            def process_data(data):
                random_search_iteractions = 10

                if isinstance(data, dict):
                    for key, value in list(data.items()):
                        # Replace None values with proper defaults
                        if value is None:
                            del data[key]
                        elif isinstance(value, dict) and 'min_int' in value and 'max_int' in value:
                            data[key] = randint(value['min_int'], value['max_int'])
                        elif isinstance(value, dict) and 'min_float' in value and 'max_float' in value:
                            data[key] = uniform(value['min_float'], value['max_float'])
                        elif isinstance(value, dict):
                            process_data(value)
                        elif key == "hidden_layer_sizes" and isinstance(value, list):
                            data[key] = [tuple(layer) for layer in value]
                        elif value == "true":
                            data[key] = True
                        elif value == "false":
                            data[key] = False
                        elif key == "random_search_iteractions":
                            random_search_iteractions = data[key]
                            del data[key]


                elif isinstance(data, list):
                    for i in range(len(data)):
                        process_data(data[i])

                return data, random_search_iteractions
            
            return process_data(data)

    def select_model(self, model, optimization):
        def find_first_class(file_path):
            with open(file_path, "r") as file:
                tree = ast.parse(file.read(), filename=file_path)
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, ast.ClassDef):
                        return node.name  # Return the name of the first class found
            return None
        def load_class_from_file(file_path, class_name):
            spec = importlib.util.spec_from_file_location("module.name", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, class_name)
        
        model_dict = {'Naive Bayes': GaussianNB(),
                      'SVM': SVC(),
                      'MLP': MLPClassifier(n_iter_no_change=500),
                      'DecisionTree': DecisionTreeClassifier(),
                      'RandomForest': RandomForestClassifier(n_jobs=-1),
                      'KNN': KNeighborsClassifier(),
                      'LogReg': LogisticRegression(),
                      'GradientBoost': GradientBoostingClassifier(),
                      'XGBoost': xgb.XGBClassifier()
                     }

        if model == "Custom AI Model":
            class_name = find_first_class("custom AI model\\custom AI model.py")
            user_model = load_class_from_file("custom AI model\\custom AI model.py", class_name)
            model_dict["Custom AI Model"] = user_model()


        if optimization == 'Grid Search':
            try:
                param_grid_dict, _ = self.load_optimization_data("configurate optimization\\grid-search configuration.json")
                if model not in param_grid_dict:
                    raise Exception
                optimized_model = GridSearchCV(estimator=model_dict[model], param_grid = param_grid_dict[model])
            except:
                param_grid_dict, _ = self.load_optimization_data("configurate optimization\\grid-search defaults.json")
                if model not in param_grid_dict:
                     optimized_model = model_dict[model]
                else:
                    optimized_model = GridSearchCV(estimator=model_dict[model], param_grid = param_grid_dict[model])


        elif optimization == 'Random Search':
            try:
                param_random_dict, random_search_iteractions  = self.load_optimization_data("configurate optimization\\random-search configuration.json")
                if model not in param_random_dict:
                    raise Exception
                optimized_model = RandomizedSearchCV(estimator = model_dict[model], param_distributions = param_random_dict[model], n_iter=random_search_iteractions)
            except:
                param_random_dict, random_search_iteractions  = self.load_optimization_data("configurate optimization\\random-search defaults.json")
                if model not in param_random_dict:
                    optimized_model = model_dict[model]
                else:
                    optimized_model = RandomizedSearchCV(estimator = model_dict[model], param_distributions = param_random_dict[model], n_iter=random_search_iteractions)
                


        elif optimization == 'None':
            optimized_model = model_dict[model]
        else:
            raise ValueError('Wrong optimization name given')

        self.get_metrics(optimized_model)

    def get_metrics(self, classifier):
        classifier.fit(self.X_train, self.y_train)
        self.y_pred = classifier.predict(self.X_test)

        # Calculate classification scores (F1, Recall, Precision and Accuracy)
        self.f1_score = f1_score(self.y_test, self.y_pred, average='weighted')
        self.recall = recall_score(self.y_test, self.y_pred, average='weighted')
        self.precision = precision_score(self.y_test, self.y_pred, average='weighted')
        self.accuracy = accuracy_score(self.y_test, self.y_pred)

        # Calculate ROC AUC
        label_binarizer = LabelBinarizer()
        y_true_binary = label_binarizer.fit_transform(self.y_test)

        if hasattr(classifier, "predict_proba"):
            y_pred_proba = classifier.predict_proba(self.X_test)
            n_classes = y_pred_proba.shape[1]

            if n_classes > 2:
                roc_auc = {}
                for i in range(n_classes):
                    for j in range(min(y_true_binary.shape[1], y_pred_proba.shape[1])):
                        try:
                            roc_auc[j] = roc_auc_score(y_true_binary[:, j], y_pred_proba[:, j])
                        except IndexError:
                            continue  # Skips if the class is missing in the current fold
                self.roc_auc = np.mean(list(roc_auc.values()))
            elif n_classes == 2:
                self.roc_auc = roc_auc_score(y_true_binary, y_pred_proba[:, 1])
            elif n_classes == 1:
                self.roc_auc = roc_auc_score(y_true_binary, y_pred_proba)
            else:
                raise ValueError("Number of classes should be at least 1 for ROC AUC calculation.")
        else:
            # For classifiers without predict_proba, use decision_function or other methods
            if hasattr(classifier, "decision_function"):
                decision_function = classifier.decision_function(self.X_test)
            else:
                decision_function = classifier.predict(self.X_test)  # fallback to predict if decision_function is not available

            # Assuming decision_function now contains scores that can be used for ROC AUC
            fpr, tpr, _ = roc_curve(y_true_binary.ravel(), decision_function.ravel())
            self.roc_auc = auc(fpr, tpr)



class MemoryMonitor:
    def __init__(self, pid, interval=0.01):
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

given_dataset = pd.read_csv('resources\\cleaned_data.csv')
given_technique = sys.argv[1]
given_model = sys.argv[2]
given_optimization = sys.argv[3]
given_cross_validation = sys.argv[4]
given_parameters = sys.argv[5]

# Getting the memory usage
current_pid = os.getpid()
memory_monitor = MemoryMonitor(current_pid)

monitor_thread = threading.Thread(target=memory_monitor)
monitor_thread.daemon = True
monitor_thread.start()

start_time = time.time()

analysis = Analysis(given_dataset, given_technique, given_optimization, given_cross_validation, given_model)

end_time = time.time()

processing_time = end_time - start_time


given_parameters = given_parameters.split(',')
result = f"{given_technique},{given_model}"
if 'F1-Score' in given_parameters:
    result += f",{analysis.f1_score}"
if 'Processing Time' in given_parameters:
    result += f",{processing_time}"
if 'ROC AUC' in given_parameters:
    result += f",{analysis.roc_auc}"
if 'Memory Usage' in given_parameters:
    result += f",{memory_monitor.max_memory_rss - memory_monitor.initial_memory_usage}"
if 'Precision' in given_parameters:
    result += f",{analysis.precision}"
if 'Accuracy' in given_parameters:
    result += f",{analysis.accuracy}"
if 'Recall' in given_parameters:
    result += f",{analysis.recall}"

print(result)