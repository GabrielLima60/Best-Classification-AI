import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import time
import sys
import tracemalloc

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

class Analysis:

    def __init__(self, dataframe, technique, model='SVM'):
        self.prepare_data(dataframe)
        self.perform_analysis(technique, model)

    def prepare_data(self, dataframe):
        # Drop 'id' column and shuffle the dataframe
        if 'id' in dataframe.columns:
            dataframe.drop('id', axis=1, inplace=True)
        dataframe = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)

        for column in dataframe.columns:
            dataframe[column] = dataframe[column].apply(lambda x: pd.to_numeric(x, errors='coerce') if isinstance(x, str) and any(char.isdigit() for char in x) else x)

        dataframe = dataframe.dropna()

        # Define X matrix and y column
        self.X = dataframe.iloc[:, :-1]
        self.y = dataframe.iloc[:, -1]

    def perform_analysis(self, technique, model):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=42)

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
        ica = FastICA(n_components=n_components, random_state=42)
        self.X_train = ica.fit_transform(self.X_train)
        self.X_test = ica.transform(self.X_test)

    def apply_lda(self):
        lda = LinearDiscriminantAnalysis()
        self.X_train = lda.fit_transform(self.X_train, self.y_train)

        self.X_test = lda.transform(self.X_test)

    def apply_smote(self):
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)  


    def apply_normalization(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def select_model_and_get_f1(self, model):
        model_dict = {'Naive Bayes': GaussianNB(),
                      'SVM': SVC(kernel='linear'),
                      'MLP': MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', max_iter=300),
                      'Tree': DecisionTreeClassifier(),
                      'KNN': KNeighborsClassifier(),
                      'LogReg': LogisticRegression(),
                      'GBC': GradientBoostingClassifier()
                     }
        

        return self.get_f1_score(model_dict[model])

    def get_f1_score(self, classifier):
        classifier.fit(self.X_train, self.y_train)
        self.y_pred = classifier.predict(self.X_test)
        return f1_score(self.y_test, self.y_pred, average='weighted')
    


# We get the parameters passed by the parent script
dataset = sys.argv[1]
dataset = pd.read_csv(dataset)
technique = sys.argv[2]
model = sys.argv[3]


# We use tracemalloc snapshots to get the memory used immediately before the execution and after it. 
tracemalloc.start()
start_time = time.time()
start_snapshot = tracemalloc.take_snapshot()


a = Analysis(dataset, technique, model)


end_snapshot = tracemalloc.take_snapshot()
end_time = time.time()
tracemalloc.stop()

# The memory usage is shown in Kibibytes (KiB)
diff_snapshot = end_snapshot.compare_to(start_snapshot, 'lineno')
memory_usage = sum(stat.size for stat in diff_snapshot)/1024

processing_time = end_time - start_time

result = str(technique) + "," + str(model) + "," + str(a.f1_score) + "," + str(processing_time) + "," + str(memory_usage)

print(result)