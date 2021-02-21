from base_data_manager import BaseDataManager


class TextClassificationDataManager(BaseDataManager):
    """Data manager for text classification"""
    def __init__(self, args, model_args, perprocessor, client_idx=None):
        super.__init__(args, model_args, perprocessor, client_idx)
        self.num_labels = -1
        self.label_vocab = None
        self.train_dataset, self.test_dataset, self.test_examples = self.load_data(
            data_file_path=model_args.data_file_path, 
            partition_file_path=model_args.partition_file_path, 
            client_idx=client_idx)
    
    def load_data(self, data_file_path, partition_file_path, client_idx=None):
        pass

    def get_data_loader(self):
        pass


class TextClassificationInputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, x0=None, y0=None, x1=None, y1=None):
        """
        Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        if x0 is None:
            self.bboxes = None
        else:
            self.bboxes = [[a, b, c, d] for a, b, c, d in zip(x0, y0, x1, y1)]