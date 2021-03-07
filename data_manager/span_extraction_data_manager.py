from data_manager.base_data_manager import BaseDataManager


class SpanExtractionDataManager(BaseDataManager):
    """Data manager for span extraction"""
    def __init__(self, args, model_args, perprocessor, client_idx=None):
        super.__init__(args, model_args, perprocessor, client_idx)

        self.train_dataset, self.test_dataset, self.test_examples = self.load_data(
            data_file_path=model_args.data_file_path, 
            partition_file_path=model_args.partition_file_path, 
            client_idx=client_idx)
    
    def load_data(self, data_file_path, partition_file_path, client_idx=None):
        pass

    def get_data_loader(self):
        pass