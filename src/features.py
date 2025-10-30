from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FeatureExtractor:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def compute_similarity(self, q1, q2):
        q1_vec = self.vectorizer.transform([q1])
        q2_vec = self.vectorizer.transform([q2])
        return cosine_similarity(q1_vec, q2_vec)[0][0]

