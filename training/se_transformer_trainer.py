#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import logging
import math
import os

import numpy as np
import sklearn
import torch
import wandb
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
)
from torch.nn import CrossEntropyLoss
from torch.cuda import amp

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

from data_preprocessing.utils.span_extraction_utils import (
    RawResult,
    RawResultExtended,
    build_examples,
    get_best_predictions,
    get_best_predictions_extended,
    to_list,
    write_predictions,
    write_predictions_extended,
)



class SpanExtractionTrainer:
    def __init__(self, args, device, model, train_dl=None, test_dl=None, tokenizer=None):
        self.args = args
        self.device = device
        self.tokenizer = tokenizer

        # set data
        self.set_data(train_dl, test_dl)

        # model
        self.model = model
    
        # training results
        self.results = {}

    def set_data(self, train_dl=None, test_dl=None):
        # Used for fedtrainer
        self.train_dl = train_dl
        self.test_dl = test_dl


    def train_model(self, device=None):
        if not device:
            device = self.device

        logging.info("train_model self.device: " + str(device))
        self.model.to(device)

        args = self.args


        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [p for n, p in self.model.named_parameters() if n in params]
            optimizer_grouped_parameters.append(param_group)

        for group in args.custom_layer_parameters:
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

        if not args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in self.model.named_parameters()
                            if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
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

        # build optimizer and scheduler
        iteration_in_total = len(
            self.train_dl) // args.gradient_accumulation_steps * args.num_train_epochs
        optimizer, scheduler = self.build_optimizer(self.model, iteration_in_total)

        if args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # training result
        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores()
        
        if args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        for epoch in range(0, args.num_train_epochs):

            self.model.train()

            for batch_idx, batch in enumerate(self.train_dl):

                batch = tuple(t.to(device) for t in batch)
                # dataset = TensorDataset(all_guid, all_input_ids, all_attention_masks, all_token_type_ids, all_cls_index, 
                # all_p_mask, all_is_impossible)

                inputs = self._get_inputs_dict(batch)

                if args.fp16:
                    with amp.autocast():
                        outputs = self.model(**inputs)
                        # model outputs are always tuple in pytorch-transformers (see doc)
                        loss = outputs[0]
                else:
                    outputs = self.model(**inputs)
                    # model outputs are always tuple in pytorch-transformers (see doc)
                    loss = outputs[0]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()

                logging.info("epoch = %d, batch_idx = %d/%d, loss = %s" % (epoch, batch_idx,
                                                                           len(self.train_dl), loss.item()))

                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

                    if args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                if args.evaluate_during_training and (
                        args.evaluate_during_training_steps > 0
                        and global_step % args.evaluate_during_training_steps == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _ = self.eval_model(epoch, global_step)
                        # for key, value in results.items():
                        #     tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results:
                            training_progress_scores[key].append(results[key])
                        # report = pd.DataFrame(training_progress_scores)
                        # report.to_csv(
                        #     os.path.join(args.output_dir, "training_progress_scores.csv"), index=False,
                        # )

                        if not best_eval_metric:
                            best_eval_metric = results[args.early_stopping_metric]
                            # self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        if best_eval_metric and args.early_stopping_metric_minimize:
                            if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                                best_eval_metric = results[args.early_stopping_metric]
                                # self.save_model(
                                #     args.best_model_dir, optimizer, scheduler, model=model, results=results
                                # )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        logging.info(f" No improvement in {args.early_stopping_metric}")
                                        logging.info(f" Current step: {early_stopping_counter}")
                                        logging.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        logging.info(f" Patience of {args.early_stopping_patience} steps reached")
                                        logging.info(" Training terminated.")
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                        )
                        else:
                            if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                                best_eval_metric = results[args.early_stopping_metric]
                                # self.save_model(
                                #     args.best_model_dir, optimizer, scheduler, model=model, results=results
                                # )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        logging.info(f" No improvement in {args.early_stopping_metric}")
                                        logging.info(f" Current step: {early_stopping_counter}")
                                        logging.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        logging.info(f" Patience of {args.early_stopping_patience} steps reached")
                                        logging.info(" Training terminated.")
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                        )

            # if args.save_model_every_epoch or args.evaluate_during_training:
            #     os.makedirs(output_dir_current, exist_ok=True)

            # if args.save_model_every_epoch:
            #     self.save_model(output_dir_current, optimizer, scheduler, model=model)

            if args.evaluate_during_training and args.evaluate_each_epoch:
                results, _, _ = self.eval_model(epoch, global_step)

                # self.save_model(output_dir_current, optimizer, scheduler, results=results)

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(os.path.join(args.output_dir, "training_progress_scores.csv"), index=False)

                if not best_eval_metric:
                    best_eval_metric = results[args.early_stopping_metric]
                    # self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                if best_eval_metric and args.early_stopping_metric_minimize:
                    if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                        best_eval_metric = results[args.early_stopping_metric]
                        # self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                logging.info(f" No improvement in {args.early_stopping_metric}")
                                logging.info(f" Current step: {early_stopping_counter}")
                                logging.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                logging.info(f" Patience of {args.early_stopping_patience} steps reached")
                                logging.info(" Training terminated.")
                                return (
                                    global_step,
                                    tr_loss / global_step
                                )
                else:
                    if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                        best_eval_metric = results[args.early_stopping_metric]
                        # self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                logging.info(f" No improvement in {args.early_stopping_metric}")
                                logging.info(f" Current step: {early_stopping_counter}")
                                logging.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                logging.info(f" Patience of {args.early_stopping_patience} steps reached")
                                logging.info(" Training terminated.")
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not args.evaluate_during_training
                                    else training_progress_scores,
                                )


        return global_step, tr_loss / global_step

    def eval_model(self, epoch=0, global_step=0, device=None):
        return {}, None, None

    def compute_metrics(self, preds, truth, eval_examples=None):
        pass

    def build_optimizer(self, model, iteration_in_total):
        warmup_steps = math.ceil(iteration_in_total * self.args.warmup_ratio)
        self.args.warmup_steps = warmup_steps if self.args.warmup_steps == 0 else self.args.warmup_steps
        logging.info("warmup steps = %d" % self.args.warmup_steps)
        # optimizer = torch.optim.Adam(self._get_optimizer_grouped_parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.01)
        optimizer = AdamW(model.parameters(), lr=self.args.learning_rate,
                          eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=iteration_in_total
        )
        return optimizer, scheduler

    def _create_training_progress_scores(self, **kwargs):
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "correct": [],
            "similar": [],
            "incorrect": [],
            "train_loss": [],
            "eval_loss": [],
            **extra_metrics,
        }

        return training_progress_scores
    
    def _get_inputs_dict(self, batch):
        inputs = {
            "input_ids": batch[1],
            "attention_mask": batch[2],
            "token_type_ids": batch[3],
            "start_positions": batch[4],
            "end_positions": batch[5],
        }

        if self.args.model_type in ["xlm", "roberta", "distilbert", "camembert", "electra", "xlmroberta", "bart"]:
            del inputs["token_type_ids"]

        if self.args.model_type in ["xlnet", "xlm"]:
            inputs.update({"cls_index": batch[6], "p_mask": batch[7]})

        return inputs