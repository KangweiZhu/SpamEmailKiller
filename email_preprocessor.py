import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import os
nltk.download('punkt')
nltk.download('stopwords')

class EmailPreprocessor:    
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        """
        TfidfVectorizer:
            feature number: 10000
                max_features: 
            docuement frequency:
                min_df: 3
                max_df: 0.9
            stop_words: use English 
            ngram_range: {1,2}
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,    
            min_df=3,             
            max_df=0.9,           
            stop_words='english', 
            ngram_range=(1, 2)    
        )
    
    def preprocess_email(self, body_text):
        if not body_text:
            return
        if not isinstance(body_text, str):
            return ''
        text = body_text.strip().lower()

        # HardCode: Just replace the url in text to url
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(),]|%[0-9a-fA-F][0-9a-fA-F])+', 'url', text)

        # Same rule apply to email
        text = re.sub(r'[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', r'email', text)

        # Replace the price related, pertenage  symbol
        text = re.sub(r'[$€£¥]', 'money', text)
        text = re.sub(r'\d+%', 'percentage', text)

        # Replace the symbol with its semantic meaning.
        text = re.sub(r'[!]{2,}', ' EXCLAMATION '.lower(), text)
        text = re.sub(r'[?]{2,}', ' question ', text)
        # other symbol just go whitespaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        """
        tokenization, remove stopwrods and remove the meaningless word like a b c d. These kind of words is largely occurred in
        """
        words = word_tokenize(text)
        words = [w for w in words if w not in self.stopwords]
        words = [w for w in words if len(w) > 1]
        return ' '.join(words)
    
    def extract_features(self, processed_emails):
        return self.tfidf_vectorizer.fit_transform(processed_emails)