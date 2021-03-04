from data_preprocessing.base.base_preprocessor import BasePreprocessor

class Preprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super(Preprocessor, self).__init__(kwargs)
    
    def transform(self, X, y, history):
        transformed_X = list()
        transformed_y = list()
        transformed_history = list()
        for i, single_x in enumerate(X):
            x_tokens = [token.text.strip().lower() for token in self.tokenizer(single_x.strip()) if token.text.strip()]
            x_token_ids = [self.word_vocab[token] if token in self.word_vocab else self.word_vocab["<UNK>"] for token in x_tokens]
            transformed_X.append(x_token_ids)
            y_tokens = [token.text.strip().lower() for token in self.tokenizer(y[i].strip()) if token.text.strip()]
            y_token_ids = [self.word_vocab[token] if token in self.word_vocab else self.word_vocab["<UNK>"] for token in y_tokens]
            transformed_y.append(y_token_ids)
            transformed_single_history = list()
            for single_h in history[i]:
                h_tokens = [token.text.strip().lower() for token in self.tokenizer(single_h.strip()) if token.text.strip()]
                h_token_ids = [self.word_vocab[token] if token in self.word_vocab else self.word_vocab["<UNK>"] for token in h_tokens]
                transformed_single_history.append(h_token_ids)
            transformed_history.append(transformed_single_history)
        return transformed_X, transformed_y, transformed_history