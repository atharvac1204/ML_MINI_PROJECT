import pickle
from sklearn.linear_model import LogisticRegression

class DuplicateModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def save(self, path='../models/model.pkl', vectorizer=None):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'vectorizer': vectorizer}, f)
        print(f"💾 Model saved at {path}")

    @staticmethod
    def load(path='../models/model.pkl'):
        with open(path, 'rb') as f:
            saved = pickle.load(f)
        print("✅ Model loaded")
        return saved['model'], saved['vectorizer']
