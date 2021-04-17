#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import logging
import math
import os

import numpy as np
import torch
import wandb
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from examples.text_classification.classification_utils import (
    InputExample,
    LazyClassificationDataset,
    convert_examples_to_features,
)
from examples.text_classification.model_args import ClassificationArgs
from model.nlp.classification.bert_model import BertForSequenceClassification

logger = logging.getLogger(__name__)


class TextClassificationTrainer:
    def __init__(self, model_type, model_name, num_labels=None, labels_map=None, args=None, **kwargs):

        """
        Initializes a ClassificationModel model.

        Args:
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert)
            model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            num_labels (optional): The number of labels or classes in the dataset.
            labels_map (optional): The mapping of the labels.
            weight (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            onnx_execution_provider (optional): ExecutionProvider to use with ONNX Runtime. Will use CUDA (if use_cuda) or CPU (if use_cuda is False) by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"MODEL_CLASSES

        # model arguments
        self.args = self._load_model_args(model_name)
        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, ClassificationArgs):
            self.args = args
        self.args.model_name = model_name
        self.args.model_type = model_type

        # data preprocessing
        len_labels_list = 2 if not num_labels else num_labels
        self.args.labels_list = [i for i in range(len_labels_list)]
        logging.info("self.args.labels_list = " + str(self.args.labels_list))
        self.num_labels = num_labels
        logging.info("num_labels = %d" % num_labels)

        # create model, tokenizer, and model config (HuggingFace style)
        MODEL_CLASSES = {
            "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
        }
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.config = config_class.from_pretrained(model_name, num_labels=num_labels, **self.args.config)
        self.model = model_class.from_pretrained(model_name, config=self.config, **kwargs)
        self.tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=self.args.do_lower_case, **kwargs)
        logging.info(self.model)

        self.device = torch.device("cuda:" + str(self.args.device_id))
        # training results
        self.results = {}
        self.best_accuracy = 0.0

    def train_model(self, train_df, eval_df=None, **kwargs):
        """
        Trains the model using 'train_df'

        Args:
            train_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be trained on this Dataframe.
        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """
        # data loading
        train_examples = [
            InputExample(oid, text, None, label)
            for i, (text, label, oid) in enumerate(zip(train_df.iloc[:, 0], train_df.iloc[:, 1], train_df.iloc[:, 2]))
        ]
        train_dataset = self.load_and_cache_examples(train_examples)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                      batch_size=self.args.train_batch_size,
                                      num_workers=self.args.dataloader_num_workers)

        # start the training loop
        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.epochs

        warmup_steps = math.ceil(t_total * self.args.warmup_ratio)
        self.args.warmup_steps = warmup_steps if self.args.warmup_steps == 0 else self.args.warmup_steps
        optimizer = torch.optim.Adam(self._get_optimizer_grouped_parameters(), lr=self.args.learning_rate, betas=(0.9, 0.999), weight_decay=0.01)
        # optimizer = AdamW(self._get_optimizer_grouped_parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        self.model.to(self.device)
        self.model.train()
        for epoch in range(self.args.epochs):

            for batch_idx, batch in enumerate(train_dataloader):
                self.model.train()
                inputs = self._get_inputs_dict(batch)
                outputs = self.model(**inputs)
                # model outputs are always tuple in pytorch-transformers (see doc)
                loss = outputs[0]
                current_loss = loss.item()
                logging.info("epoch = %d, batch_idx = %d/%d, loss = %s" % (epoch, batch_idx,
                                                                           len(train_dataloader), current_loss))

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        if self.args.wandb_project:
                            wandb.log({"Training loss": current_loss,
                                       "lr": scheduler.get_last_lr()[0], "global_step": global_step})

                    if self.args.evaluate_during_training and (self.args.evaluate_during_training_steps > 0
                            and global_step % self.args.evaluate_during_training_steps == 0):
                        results, _, _ = self.eval_model(eval_df,
                                                        silent=self.args.evaluate_during_training_silent,
                                                        **kwargs)

        return global_step, tr_loss / global_step

    def eval_model(self, eval_df, silent=False, **kwargs):
        """
        Evaluates the model on eval_df. Saves results to output_dir.

        Args:
            eval_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be evaluated on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            silent: If silent, tqdm progress bars will be hidden.
            wandb_log: If True, evaluation results will be logged to wandb.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results.
            model_outputs: List of model outputs for each row in eval_df
            wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model
        """  # noqa: ignore flake8"
        self.model.to(self.device)
        model = self.model
        args = self.args

        results = {}

        eval_examples = [
            InputExample(i, text, None, label)
            for i, (text, label) in enumerate(zip(eval_df.iloc[:, 0], eval_df.iloc[:, 1]))
        ]
        eval_dataset = self.load_and_cache_examples(
            eval_examples, evaluate=True, silent=silent
        )

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(eval_dataloader)
        preds = np.empty((len(eval_dataset), self.num_labels))
        out_label_ids = np.empty((len(eval_dataset)))
        model.eval()
        for i, batch in enumerate(tqdm(eval_dataloader, disable=args.silent or silent,
                                       desc="{kwargs['client_desc']} :| Running Evaluation")):
            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1
            start_index = self.args.eval_batch_size * i
            end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else len(eval_dataset)
            preds[start_index:end_index] = logits.detach().cpu().numpy()
            out_label_ids[start_index:end_index] = inputs["labels"].detach().cpu().numpy()

        eval_loss = eval_loss / nb_eval_steps

        model_outputs = preds
        preds = np.argmax(preds, axis=1)
        result, wrong = self.compute_metrics(preds, out_label_ids, eval_examples, **kwargs)
        result["eval_loss"] = eval_loss
        results.update(result)

        os.makedirs(self.args.output_dir, exist_ok=True)
        output_eval_file = os.path.join(self.args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        if result["acc"] > self.best_accuracy:
            self.best_accuracy = result["acc"]
        logging.info("best_accuracy = %f" % self.best_accuracy)
        self.results.update(result)
        logger.info(self.results)

        return result, model_outputs, wrong

    def load_and_cache_examples(self, examples, evaluate=False, no_cache=False, multi_label=False, silent=False):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """
        process_count = self.args.process_count

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not multi_label and args.regression:
            output_mode = "regression"
        else:
            output_mode = "classification"

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"
        cached_features_file = os.path.join(
            args.cache_dir,
            "cached_{}_{}_{}_{}_{}".format(
                mode, args.model_type, args.max_seq_length, self.num_labels, len(examples),
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
            logger.info(f" Features loaded from cache at {cached_features_file}")
        else:
            logger.info(" Converting to features started. Cache is not used.")

            # If labels_map is defined, then labels need to be replaced with ints
            if self.args.labels_map and not self.args.regression:
                for example in examples:
                    if multi_label:
                        example.label = [self.args.labels_map[label] for label in example.label]
                    else:
                        example.label = self.args.labels_map[example.label]

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
                multi_label=multi_label,
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
            logger.info(f" {len(features)} features created from {len(examples)} samples.")

            if not no_cache:
                torch.save(features, cached_features_file)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        return dataset

    def compute_metrics(self, preds, labels, eval_examples=None, **kwargs):
        """
        Computes the evaluation metrics for the model predictions.

        Args:
            preds: Model predictions
            labels: Ground truth labels
            eval_examples: List of examples on which evaluation was performed
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)
            wrong: List of InputExample objects corresponding to each incorrect prediction by the model
        """  # noqa: ignore flake8"

        assert len(preds) == len(labels)

        extra_metrics = {}
        for metric, func in kwargs.items():
            extra_metrics[metric] = func(labels, preds)

        mismatched = labels != preds

        if eval_examples:
            wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        else:
            wrong = ["NA"]

        mcc = matthews_corrcoef(labels, preds)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        return (
            {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics},
            wrong,
        )

    def _get_inputs_dict(self, batch):
        if isinstance(batch[0], dict):
            inputs = {key: value.squeeze().to(self.device) for key, value in batch[0].items()}
            inputs["labels"] = batch[1].to(self.device)
        else:
            batch = tuple(t.to(self.device) for t in batch)

            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

            # XLM, DistilBERT and RoBERTa don't use segment_ids
            if self.args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if self.args.model_type in ["bert", "xlnet", "albert", "layoutlm"] else None
                )

        return inputs

    def _get_optimizer_grouped_parameters(self):
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [p for n, p in self.model.named_parameters() if n in params]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in self.model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in self.model.named_parameters()
                            if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in self.model.named_parameters()
                            if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )
        return optimizer_grouped_parameters

    def save_model(self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None):
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            self.save_model_args(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = ClassificationArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]
