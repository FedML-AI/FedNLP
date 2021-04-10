from data_manager.base_data_manager import BaseDataManager
import h5py
from torch.utils.data import DataLoader
import logging
import numpy as np

class SequenceTaggingDataManager(BaseDataManager):
    """Data manager for sequence tagging tasks."""
    def __init__(self, args, model_args, preprocessor, process_id=0, num_workers=1):
        # TODO: ref to a defination of the "args" and "model_args"
        #           --- what will they must contain? (e.g., data_file_path)

        super(SequenceTaggingDataManager, self).__init__(args, model_args, process_id, num_workers)
        self.attributes = self.load_attributes(args.data_file_path)
        self.preprocessor = preprocessor

        # TODO: maybe add this back, currently as the external usage
        # self.load_next_round_data()
        # self.train_loader, self.test_loader = self.get_data_loader()


    def read_instance_from_h5(self, data_file, index_list):
        X = [[s.decode("utf-8") for s in data_file["X"][str(idx)][()]] for idx in index_list]
        y = [[s.decode("utf-8") for s in data_file["Y"][str(idx)][()]] for idx in index_list]
        return {"X": X, "y": y}