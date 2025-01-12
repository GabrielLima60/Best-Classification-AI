import numpy as np

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_prior = None
        self.mean = None
        self.var = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.class_prior = np.zeros(n_classes, dtype=np.float64)

        for idx, cls in enumerate(self.classes):
            X_c = X[y == cls]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.class_prior[idx] = X_c.shape[0] / float(n_samples)

    def _gaussian_probability(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-0.5 * ((x - mean) ** 2) / (var + 1e-9))
        denominator = np.sqrt(2 * np.pi * var + 1e-9)
        return numerator / denominator

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probs = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            for idx, cls in enumerate(self.classes):
                class_prior = np.log(self.class_prior[idx])
                conditional = np.sum(np.log(self._gaussian_probability(idx, X[i])))
                probs[i, idx] = class_prior + conditional

        return np.exp(probs) / np.sum(np.exp(probs), axis=1, keepdims=True)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]