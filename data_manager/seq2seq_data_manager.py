from data_manager.base_data_manager import BaseDataManager


class Seq2SeqDataManager(BaseDataManager):
    """Data manager for seq2seq"""
    def __init__(self, args, model_args, preprocessor, process_id=0, num_workers=1):
        # TODO: ref to a defination of the "args" and "model_args"
        #           --- what will they must contain? (e.g., data_file_path)

        super(Seq2SeqDataManager, self).__init__(args, model_args, process_id, num_workers)
        self.attributes = self.load_attributes(args.data_file_path)
        self.preprocessor = preprocessor

        # TODO: maybe add this back, currently as the external usage
        # self.load_next_round_data()
        # self.train_loader, self.test_loader = self.get_data_loader()


    def read_instance_from_h5(self, data_file, index_list, desc=""):
        X = [data_file["X"][str(idx)][()].decode("utf-8") for idx in index_list]
        y = [data_file["Y"][str(idx)][()].decode("utf-8") for idx in index_list]
        return X, y