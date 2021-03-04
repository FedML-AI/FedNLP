from data_preprocessing.base.base_preprocessor import BasePreprocessor
import re

class Preprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super(Preprocessor, self).__init__(kwargs)
    
    def transform(self, context_X, question_X, y):
        transformed_context_X = list()
        transformed_question_X = list()
        transformed_y = list()
        for i in range(len(context_X)):
            context = context_X[i]
            temp_tokens = self.word_tokenize(context)
            context_tokens = self.remove_stop_tokens(temp_tokens)
            question_tokens = self.word_tokenize(question_X[i])

            context_token_ids = [self.word_vocab[token] if token in self.word_vocab else self.word_vocab["<UNK>"] for token in context_tokens]
            question_token_ids = [self.word_vocab[token] if token in self.word_vocab else self.word_vocab["<UNK>"] for token in question_tokens]
            transformed_context_X.append(context_token_ids)
            transformed_question_X.append(question_token_ids)
            
            start, stop = y[i]

            spans = self.get_spans(context, context_tokens)
            idxs = []
            for word_idx, span in enumerate(spans):
                if not (stop <= span[0] or start >= span[1]):
                    idxs.append(word_idx)
            
            transformed_y.append(idxs[0], idxs[-1] + 1)
        
        return transformed_context_X, transformed_question_X, transformed_y
    
    def word_tokenize(self, sentence):
        return [token.replace("''", '"').replace("``", '"') for token in self.tokenizer.word_tokenize(sentence)]
    
    def remove_stop_tokens(self, temp_tokens):
        tokens = []
        for token in temp_tokens:
            flag = False
            l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
            tokens.extend(re.split("([{}])".format("".join(l)), token))
        return tokens
    
    def transform_labels(self, context_X, tokenized_context_X, y):
        transformed_y = list()
        for i in range(len(context_X)):
            context = context_X[i]
            context_tokens = tokenized_context_X[i]
            start, stop = y[i]

            spans = self.get_spans(context, context_tokens)
            idxs = []
            for word_idx, span in enumerate(spans):
                if not (stop <= span[0] or start >= span[1]):
                    idxs.append(word_idx)
            
            transformed_y.append(idxs[0], idxs[-1] + 1)
        return transformed_y

    def get_spans(self, text, all_tokens):
        spans = []
        cur_idx = 0
        for token in all_tokens:
            if text.find(token, cur_idx) < 0:
                print("{} {} {}".format(token, cur_idx, text))
                raise Exception()
            cur_idx = text.find(token, cur_idx)
            spans.append((cur_idx, cur_idx + len(token)))
            cur_idx += len(token)
        return spans