import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

class TextPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean(self, text: str) -> str:
        # lowercase
        if self.config.get('lowercase', True):
            text = text.lower()
        # remove URLs
        if self.config.get('remove_urls', True):
            text = re.sub(r'http\S+|www\S+', '', text)
        # remove HTML tags
        if self.config.get('remove_html', True):
            text = re.sub(r'<.*?>', '', text)
        # remove punctuation
        if self.config.get('remove_punctuation', True):
            text = text.translate(str.maketrans('', '', string.punctuation))
        # remove numbers
        if self.config.get('remove_numbers', False):
            text = re.sub(r'\d+', '', text)
        return text.strip()

    def tokenize(self, text: str) -> list:
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list) -> list:
        if self.config.get('remove_stopwords', True):
            return [t for t in tokens if t not in self.stop_words]
        return tokens

    def stem(self, tokens: list) -> list:
        if self.config.get('stemming', False):
            return [self.stemmer.stem(t) for t in tokens]
        return tokens

    def lemmatize(self, tokens: list) -> list:
        if self.config.get('lemmatization', False):
            return [self.lemmatizer.lemmatize(t) for t in tokens]
        return tokens

    def process(self, text: str) -> str:
        text = self.clean(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem(tokens)
        tokens = self.lemmatize(tokens)
        return ' '.join(tokens)