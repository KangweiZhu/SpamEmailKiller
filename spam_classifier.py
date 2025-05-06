from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import re

class BaselineSpamClassifier:
    
    def __init__(self):
        with open('spam_keywords.txt', 'r', encoding='utf-8') as f:
            self.spam_keywords = set(line.strip().lower() for line in f) 
    
    def predict(self, email):
        # If contains 3 or more keywords, it is considered spam
        email = email.lower()
        spam_score = sum(1 for keyword in self.spam_keywords if keyword in email)
        return 1 if spam_score >= 3 else 0

class MLSpamClassifier:
    def __init__(self, classifier_type='svm'):
        if classifier_type == 'nb':
            self.classifier = MultinomialNB(alpha=0.1)
        elif classifier_type == 'svm':
            self.classifier = LinearSVC(random_state=40, class_weight='balanced')
        self.classifier_type = classifier_type
    
    def train(self, X, y):
        self.classifier.fit(X, y)
    
    def predict(self, X):
        return self.classifier.predict(X)
    
    def score(self, X, y):
        return self.classifier.score(X, y) 