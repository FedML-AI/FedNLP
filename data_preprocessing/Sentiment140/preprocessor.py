from data_preprocessing.base.base_preprocessor import BasePreprocessor
import re

class Preprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super(Preprocessor, self).__init__(kwargs)
    
    def transform(self, X, y):
        transformed_X = list()
        transformed_y = list()
        for i, single_x in enumerate(X):
            cleaned_single_x = self.clean_sentence(single_x)
            x_tokens = [token.text.strip().lower() for token in self.tokenizer(cleaned_single_x.strip()) if token.text.strip()]
            x_token_ids = [self.word_vocab[token] if token in self.word_vocab else self.word_vocab["<UNK>"] for token in x_tokens]
            transformed_X.append(x_token_ids)
            transformed_y.append(self.label_vocab[y[i]])
        return transformed_X, transformed_y
    
    def clean_sentence(self, sentence):
        sentence = re.sub(r'\&\w*;', '', sentence)
        sentence = re.sub('@[^\s]+','',sentence)
        sentence = re.sub(r'\$\w*', '', sentence)
        sentence = sentence.lower()
        sentence = re.sub(r'https?:\/\/.*\/\w*', '', sentence)
        sentence = re.sub(r'#\w*', '', sentence)
        sentence = re.sub(r'[' + string.punctuation.replace('@', '') + ']+', ' ', sentence)
        sentence = re.sub(r'\b\w{1,2}\b', '', sentence)
        sentence = re.sub(r'\s\s+', ' ', sentence)
        sentence = [char for char in list(sentence) if char not in string.punctuation]
        sentence = ''.join(sentence)
        sentence = sentence.lstrip(' ') 
        return sentence