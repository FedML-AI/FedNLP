import logging

import torch
import pandas as pd
from FedML.fedml_core.trainer.model_trainer import ModelTrainer
from model.fed_transformers.classification.classification_utils import (
    InputExample,
    LazyClassificationDataset
)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import warnings
import os

class TransformerTrainer(ModelTrainer):

    def __init__(self, transformer_model):
        self.transformer_model = transformer_model
        self.model = self.transformer_model.model
        self.id = 0

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train_model(
        self,
        train_df,
        multi_label=False,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_df=None,
        verbose=True,
        **kwargs,
    ):
        if args:
            self.transformer_model.args.update_from_dict(args)

        if self.transformer_model.args.silent:
            show_running_loss = False

        if self.transformer_model.args.evaluate_during_training and eval_df is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_df is not specified."
                " Pass eval_df to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.transformer_model.args.output_dir

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.transformer_model.args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set overwrite_output_dir: True to automatically overwrite.".format(output_dir)
            )    

        if isinstance(train_df, str) and self.transformer_model.args.lazy_loading:
            if self.transformer_model.args.sliding_window:
                raise ValueError("Lazy loading cannot be used with sliding window.")
            if self.transformer_model.args.model_type == "layoutlm":
                raise NotImplementedError("Lazy loading is not implemented for LayoutLM models")
            train_dataset = LazyClassificationDataset(train_df, self.transformer_model.tokenizer, self.transformer_model.args)
        else:
            if self.transformer_model.args.lazy_loading:
                raise ValueError("Input must be given as a path to a file when using lazy loading")
            if "text" in train_df.columns and "labels" in train_df.columns:
                # The most commonly used way of building examples.
                train_examples = [
                    InputExample(i, text, None, label)
                    for i, (text, label) in enumerate(zip(train_df["text"].astype(str), train_df["labels"]))
                ]
            elif "text_a" in train_df.columns and "text_b" in train_df.columns:
                if self.transformer_model.args.model_type == "layoutlm":
                    raise ValueError("LayoutLM cannot be used with sentence-pair tasks")
                else:
                    train_examples = [
                        InputExample(i, text_a, text_b, label)
                        for i, (text_a, text_b, label) in enumerate(
                            zip(train_df["text_a"].astype(str), train_df["text_b"].astype(str), train_df["labels"])
                        )
                    ]
            else:
                warnings.warn(
                    "Dataframe headers not specified. Falling back to using column 0 as text and column 1 as labels."
                )
                train_examples = [
                    InputExample(i, text, None, label)
                    for i, (text, label) in enumerate(zip(train_df.iloc[:, 0], train_df.iloc[:, 1]))
                ]
            train_dataset = self.transformer_model.load_and_cache_examples(train_examples, verbose=verbose)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.transformer_model.args.train_batch_size,
            num_workers=self.transformer_model.args.dataloader_num_workers,
        )

        os.makedirs(output_dir, exist_ok=True)

        global_step, training_details = self.transformer_model.train(
            train_dataloader,
            output_dir,
            multi_label=multi_label,
            show_running_loss=show_running_loss,
            eval_df=eval_df,
            verbose=verbose,
            **kwargs,
        )

        # model_to_save = self.transformer_model.model.module if hasattr(self.transformer_model.model, "module") else self.transformer_model.model
        # model_to_save.save_pretrained(output_dir)
        # self.transformer_model.tokenizer.save_pretrained(output_dir)
        # torch.save(self.transformer_model.args, os.path.join(output_dir, "training_args.bin"))
        self.transformer_model.save_model(model=self.transformer_model.model)

        if verbose:
            logging.info(" Training of {} model complete. Saved to {}.".format(self.transformer_model.args.model_type, output_dir))

        return global_step, training_details


    def train(self, train_data, device, args):
        self.device = device
        self.transformer_model._move_model_to_device() 
        labels_map = self.transformer_model.labels_map
        train_data = [(x, labels_map[y])
                  for x, y in zip(train_data["X"], train_data["Y"])]
        train_df = pd.DataFrame(train_data)
        global_step, training_details = self.train_model(train_df=train_df)

    def test(self, test_data, device, args):
        # TODO:
        # return test_acc, test_total, test_loss
        # self.transformer_model.eval_model()
        return None

    def test_on_the_server(self, test_data, device, args):
        # TODO:
        # return test_acc, test_total, test_loss
        # self.transformer_model.eval_model()
        return None
    
