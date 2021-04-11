from data_manager.base_data_manager import BaseDataManager
from tqdm import tqdm

class SpanExtractionDataManager(BaseDataManager):
    """Data manager for reading comprehension (span-based QA).""" 
    def __init__(self, args, model_args, preprocessor, process_id=0, num_workers=1):
        # TODO: ref to a defination of the "args" and "model_args"
        #           --- what will they must contain? (e.g., data_file_path)
        super(SpanExtractionDataManager, self).__init__(args, model_args, process_id, num_workers)
        self.attributes = self.load_attributes(args.data_file_path)
        self.preprocessor = preprocessor

        
    def read_instance_from_h5(self, data_file, index_list):
        context_X = list()
        question_X = list()
        y = list()
        for idx in tqdm(index_list, desc="Loading data from h5 file."):
            context_X.append(data_file["context_X"][str(idx)][()].decode("utf-8"))
            question_X.append(data_file["question_X"][str(idx)][()].decode("utf-8"))
            y.append(data_file["Y"][str(idx)][()] )
        return {"context_X": context_X, "question_X": question_X, "y": y}