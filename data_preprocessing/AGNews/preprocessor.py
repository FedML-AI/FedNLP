from data_preprocessing.base.base_preprocessor import BasePreprocessor
from data_preprocessing.base.base_example import TextClassificationInputExample
from model.fed_transformers.classification.classification_utils import *
from torch.utils.data import DataLoader, TensorDataset
import torch
import logging
import pandas as pd

class BertPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super(BertPreprocessor, self).__init__(**kwargs)
    
    def transform(self, X, y, index_list=None, evaluate=False):
        if index_list is None:
            index_list = [i for i in range(len(X))]
        examples = self.transform_examples(X, y, index_list)
        features = self.transform_features(examples, evaluate=evaluate)

        all_guid = torch.tensor([f.guid for f in features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(all_guid, all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        return examples, dataset
        
    
    def transform_examples(self, X, y, index_list):
        data = [(X[i], self.label_vocab[y[i]], idx) for i, idx in enumerate(index_list)]

        df = pd.DataFrame(data)
        examples = [
            TextClassificationInputExample(guid, text, None, label)
            for i, (text, label, guid) in enumerate(zip(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]))
        ]
        
        return examples
        
    def transform_features(self, examples, evaluate=False, no_cache=False, silent=False):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.
        """
        process_count = self.args.process_count

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        output_mode = "classification"

        if not no_cache:
            os.makedirs(args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"
        cached_features_file = os.path.join(
            args.cache_dir,
            "cached_{}_{}_{}_{}_{}".format(
                mode, args.model_type, args.max_seq_length, len(self.label_vocab), len(examples),
            ),
        )
        logging.info("cached_features_file = %s" % str(cached_features_file))
        logging.info("args.reprocess_input_data = %s" % str(args.reprocess_input_data))
        logging.info("no_cache = %s" % str(no_cache))
        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not no_cache)
        ):
            features = torch.load(cached_features_file)
            logging.info(f" Features loaded from cache at {cached_features_file}")
        else:
            logging.info(" Converting to features started. Cache is not used.")

            # If labels_map is defined, then labels need to be replaced with ints
            if args.labels_map and not args.regression:
                for example in examples:
                    example.label = args.labels_map[example.label]

            features = convert_examples_to_features(
                examples,
                args.max_seq_length,
                tokenizer,
                output_mode,
                # XLNet has a CLS token at the end
                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                # RoBERTa uses an extra separator b/w pairs of sentences,
                # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                sep_token_extra=bool(args.model_type in ["roberta", "camembert", "xlmroberta", "longformer"]),
                # PAD on the left for XLNet
                pad_on_left=bool(args.model_type in ["xlnet"]),
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                process_count=process_count,
                multi_label=False,
                silent=args.silent or silent,
                use_multiprocessing=args.use_multiprocessing,
                sliding_window=args.sliding_window,
                flatten=not evaluate,
                stride=args.stride,
                add_prefix_space=bool(args.model_type in ["roberta", "camembert", "xlmroberta", "longformer"]),
                # avoid padding in case of single example/online inferencing to decrease execution time
                pad_to_max_length=bool(len(examples) > 1),
                args=args,
            )
            logging.info(f" {len(features)} features created from {len(examples)} samples.")

            if not no_cache:
                torch.save(features, cached_features_file)
        return features