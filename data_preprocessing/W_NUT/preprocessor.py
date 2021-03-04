from data_preprocessing.base.base_preprocessor import BasePreprocessor

class Preprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super(Preprocessor, self).__init__(kwargs)
    
    def transform(self, X, y):
        transformed_X = list()
        transformed_y = list()
        for i, single_x in enumerate(X):
            x_token_ids = [self.word_vocab[token] if token in self.word_vocab else self.word_vocab["<UNK>"] for token in single_x]
            transformed_X.append(x_token_ids)
            y_label_ids = [self.label_vocab[label] for label in y[i]]
            transformed_y.append(y_label_ids)
        return transformed_X, transformed_y
    