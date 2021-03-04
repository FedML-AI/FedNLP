from data_preprocessing.base.base_preprocessor import BasePreprocessor

class Preprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super(Preprocessor, self).__init__(kwargs)
    
    def transform(self, X, y):
        transformed_X = list()
        transformed_y = list()
        for i, single_x in enumerate(X):
            tokens = [token.text.strip().lower() for token in self.tokenizer(single_x.strip()) if token.text.strip()]
            token_ids = [self.word_vocab[token] if token in self.word_vocab else self.word_vocab["<UNK>"] for token in tokens]
            transformed_X.append(token_ids)
            transformed_y.append(self.label_vocab[y[i]])
        return transformed_X, transformed_y