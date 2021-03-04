from data_preprocessing.base.base_preprocessor import BasePreprocessor
import re

class Preprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super(Preprocessor, self).__init__(kwargs)
    
    def transform(self, X, y):
        transformed_X = list()
        transformed_y = list()
        for i, single_x in enumerate(X):
            cleaned_single_x = self.clean_and_tokenize_sentence(single_x)
            x_tokens = [token.text.strip().lower() for token in self.tokenizer(cleaned_single_x.strip()) if token.text.strip()]
            x_token_ids = [self.word_vocab[token] if token in self.word_vocab else self.word_vocab["<UNK>"] for token in x_tokens]
            transformed_X.append(x_token_ids)
            transformed_y.append(self.label_vocab[y[i]])
        return transformed_X, transformed_y
    
    def clean_and_tokenize_sentence(self, sentence):
        e1 = re.findall(r'<e1>(.*)</e1>', sentence)[0]
        e2 = re.findall(r'<e2>(.*)</e2>', sentence)[0]
        sentence = sentence.replace('<e1>' + e1 + '</e1>', ' <e1> ' + e1 + ' </e1> ', 1)
        sentence = sentence.replace('<e2>' + e2 + '</e2>', ' <e2> ' + e2 + ' </e2> ', 1)
        sentence = [token.text.strip() for token in self.tokenizer(sentence) if token.text.strip()]
        sentence = ' '.join(sentence)
        sentence = sentence.replace('< e1 >', '<e1>')
        sentence = sentence.replace('< e2 >', '<e2>')
        sentence = sentence.replace('< /e1 >', '</e1>')
        sentence = sentence.replace('< /e2 >', '</e2>')
        return sentence.split()