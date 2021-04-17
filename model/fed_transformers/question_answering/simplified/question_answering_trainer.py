from __future__ import absolute_import, division, print_function

import json
import logging
import math
import os

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from examples.question_answering.model_args import QuestionAnsweringArgs
from examples.question_answering.question_answering_utils import (
    RawResult,
    RawResultExtended,
    get_examples,
    squad_convert_examples_to_features,
    to_list,
    write_predictions,
    write_predictions_extended,
)

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)


class QuestionAnsweringTrainer:
    def __init__(self, model_type, model_name, args=None, **kwargs):
        """
        Initializes a QuestionAnsweringModel model.

        Args:
            model_type: The type of model (bert, xlnet, xlm, distilbert)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            args (optional): Default args will be used if this parameter is not provided. If provided,
                it should be a dict containing the args that should be changed in the default args'
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
        """  # noqa: ignore flake8"
        self.args = self._load_model_args(model_name)
        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, QuestionAnsweringArgs):
            self.args = args

        self.args.model_name = model_name
        self.args.model_type = model_type

        MODEL_CLASSES = {
            "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
        }

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.config = config_class.from_pretrained(model_name, **self.args.config)
        self.model = model_class.from_pretrained(model_name, config=self.config, **kwargs)
        self.tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=self.args.do_lower_case, **kwargs)

        logging.info(self.model)

        self.device = torch.device("cuda:2")

        self.results = {}

    def train_model(
            self, train_data, eval_data, output_dir=False, args=None, **kwargs
    ):
        """
        Trains the model using 'train_data'

        Args:
            train_data: Path to JSON file containing training data OR list of Python dicts in the correct format. The model will be trained on this data.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_data (optional): Path to JSON file containing evaluation data against which evaluation will be performed when evaluate_during_training is enabled.
                Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"
        if args:
            self.args.update_from_dict(args)

        self._move_model_to_device()

        if isinstance(train_data, str):
            with open(train_data, "r", encoding=self.args.encoding) as f:
                train_examples = json.load(f)
        else:
            train_examples = train_data

        train_dataset = self.load_and_cache_examples(train_examples)

        if not output_dir:
            output_dir = self.args.output_dir
        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                "Use --overwrite_output_dir to overcome.".format(output_dir)
            )
        os.makedirs(output_dir, exist_ok=True)

        global_step, training_details = self.train(
            train_dataset, eval_data=eval_data, **kwargs
        )
        logger.info(" Training of {} model complete. Saved to {}.".format(self.args.model_type, output_dir))

        return global_step, training_details

    def train(self, train_dataset, eval_data=None, **kwargs):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """
        device = self.device
        model = self.model
        args = self.args

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps

        optimizer = AdamW(self._get_optimizer_grouped_parameters(), lr=args.learning_rate, eps=args.adam_epsilon, )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        for epoch in range(self.args.epochs):
            model.train()
            for batch_idx, batch in enumerate(train_dataloader):
                model.train()

                batch = tuple(t.to(device) for t in batch)

                inputs = self._get_inputs_dict(batch)

                outputs = model(**inputs)
                # model outputs are always tuple in pytorch-transformers (see doc)
                loss = outputs[0]

                current_loss = loss.item()
                logging.info("epoch = %d, batch_idx = %d/%d, loss = %s" % (epoch, batch_idx,
                                                                           len(train_dataloader), current_loss))
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if self.args.evaluate_during_training and (self.args.evaluate_during_training_steps > 0
                            and global_step % self.args.evaluate_during_training_steps == 0):

                        results, _ = self.eval_model(eval_data, **kwargs)

        return global_step, tr_loss / global_step

    def eval_model(self, eval_data, **kwargs):
        """
        Evaluates the model on eval_data. Saves results to output_dir.

        Args:
            eval_data: Path to JSON file containing evaluation data OR list of Python dicts in the correct format. The model will be evaluated on this data.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            verbose_logging: Log info related to feature conversion and writing predictions.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results. (correct, similar, incorrect)
            text: A dictionary containing the 3 dictionaries correct_text, similar_text (the predicted answer is a substring of the correct answer or vise versa), incorrect_text.
        """  # noqa: ignore flake8"

        output_dir = self.args.output_dir

        self._move_model_to_device()

        all_predictions, all_nbest_json, scores_diff_json, eval_loss = self.evaluate(eval_data, output_dir)

        if isinstance(eval_data, str):
            with open(eval_data, "r", encoding=self.args.encoding) as f:
                truth = json.load(f)
        else:
            truth = eval_data

        result, texts = self.calculate_results(truth, all_predictions, **kwargs)
        result["eval_loss"] = eval_loss

        self.results.update(result)

        logger.info(self.results)

        return result, texts

    def eval_model_by_offical_script(self, eval_data, eval_data_path, output_dir=None, verbose=False,
                                     verbose_logging=False, **kwargs):
        """
        Evaluates the model on eval_data. Saves results to output_dir.

        Args:
            eval_data: list of Python dicts in the correct format. The model will be evaluated on this data.
            eval_data_path: Path to JSON file containing evaluation data
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            verbose_logging: Log info related to feature conversion and writing predictions.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results. (exact match score, F1 score)
        """
        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()

        all_predictions, all_nbest_json, scores_diff_json, eval_loss = self.evaluate(eval_data, output_dir)

        prediction_dict = dict()
        for i, prediction in all_predictions.items():
            qid = eval_data[int(i) - 1]["qas"][0]["id"]
            prediction_dict[qid] = prediction

        with open(os.path.join(output_dir, "prediction.json"), "w") as f:
            json.dump(prediction_dict, f)

        f = os.popen("python ./evaluate-v1.1.py %s %s" % (
            eval_data_path, os.path.join(output_dir, "prediction.json")))

        result = eval(f.read().strip())
        return result

    def evaluate(self, eval_data, output_dir, verbose_logging=False):
        """
        Evaluates the model on eval_data.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """
        tokenizer = self.tokenizer
        device = self.device
        model = self.model
        args = self.args

        if isinstance(eval_data, str):
            with open(eval_data, "r", encoding=self.args.encoding) as f:
                eval_examples = json.load(f)
        else:
            eval_examples = eval_data

        eval_dataset, examples, features = self.load_and_cache_examples(
            eval_examples, evaluate=True, output_examples=True
        )

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()

        all_results = []
        for batch in tqdm(eval_dataloader, disable=args.silent, desc="Running Evaluation"):
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                if self.args.model_type in [
                    "xlm",
                    "roberta",
                    "distilbert",
                    "camembert",
                    "electra",
                    "xlmroberta",
                    "bart",
                ]:
                    del inputs["token_type_ids"]

                example_indices = batch[3]

                if args.model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[4], "p_mask": batch[5]})

                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        eval_loss += outputs[0].mean().item()
                else:
                    outputs = model(**inputs)
                    eval_loss += outputs[0].mean().item()

                for i, example_index in enumerate(example_indices):
                    eval_feature = features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    if args.model_type in ["xlnet", "xlm"]:
                        # XLNet uses a more complex post-processing procedure
                        result = RawResultExtended(
                            unique_id=unique_id,
                            start_top_log_probs=to_list(outputs[0][i]),
                            start_top_index=to_list(outputs[1][i]),
                            end_top_log_probs=to_list(outputs[2][i]),
                            end_top_index=to_list(outputs[3][i]),
                            cls_logits=to_list(outputs[4][i]),
                        )
                    else:
                        result = RawResult(
                            unique_id=unique_id,
                            start_logits=to_list(outputs[0][i]),
                            end_logits=to_list(outputs[1][i]),
                        )
                    all_results.append(result)

            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps

        prefix = "test"
        os.makedirs(output_dir, exist_ok=True)

        output_prediction_file = os.path.join(output_dir, "predictions_{}.json".format(prefix))
        output_nbest_file = os.path.join(output_dir, "nbest_predictions_{}.json".format(prefix))
        output_null_log_odds_file = os.path.join(output_dir, "null_odds_{}.json".format(prefix))

        if args.model_type in ["xlnet", "xlm"]:
            # XLNet uses a more complex post-processing procedure
            (all_predictions, all_nbest_json, scores_diff_json, out_eval) = write_predictions_extended(
                examples,
                features,
                all_results,
                args.n_best_size,
                args.max_answer_length,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                eval_data,
                model.config.start_n_top,
                model.config.end_n_top,
                True,
                tokenizer,
                verbose_logging,
            )
        else:
            all_predictions, all_nbest_json, scores_diff_json = write_predictions(
                examples,
                features,
                all_results,
                args.n_best_size,
                args.max_answer_length,
                False,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                verbose_logging,
                True,
                args.null_score_diff_threshold,
            )

        return all_predictions, all_nbest_json, scores_diff_json, eval_loss


    def calculate_results(self, truth, predictions, **kwargs):
        truth_dict = {}
        questions_dict = {}
        for item in truth:
            for answer in item["qas"]:
                if answer["answers"]:
                    truth_dict[answer["id"]] = answer["answers"][0]["text"]
                else:
                    truth_dict[answer["id"]] = ""
                questions_dict[answer["id"]] = answer["question"]

        correct = 0
        incorrect = 0
        similar = 0
        correct_text = {}
        incorrect_text = {}
        similar_text = {}
        predicted_answers = []
        true_answers = []

        for q_id, answer in truth_dict.items():
            predicted_answers.append(predictions[q_id])
            true_answers.append(answer)
            if predictions[q_id].strip() == answer.strip():
                correct += 1
                correct_text[q_id] = answer
            elif predictions[q_id].strip() in answer.strip() or answer.strip() in predictions[q_id].strip():
                similar += 1
                similar_text[q_id] = {
                    "truth": answer,
                    "predicted": predictions[q_id],
                    "question": questions_dict[q_id],
                }
            else:
                incorrect += 1
                incorrect_text[q_id] = {
                    "truth": answer,
                    "predicted": predictions[q_id],
                    "question": questions_dict[q_id],
                }

        extra_metrics = {}
        logging.info(kwargs)
        for metric, func in kwargs.items():
            extra_metrics[metric] = func(true_answers, predicted_answers)

        result = {"correct": correct, "similar": similar, "incorrect": incorrect, **extra_metrics}

        texts = {
            "correct_text": correct_text,
            "similar_text": similar_text,
            "incorrect_text": incorrect_text,
        }

        return result, texts

    def load_and_cache_examples(self, examples, evaluate=False, no_cache=False, output_examples=False):
        """
        Converts a list of examples to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """
        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        examples = get_examples(examples, is_training=not evaluate)

        mode = "dev" if evaluate else "train"
        cached_features_file = os.path.join(
            args.cache_dir, "cached_{}_{}_{}_{}".format(mode, args.model_type, args.max_seq_length, len(examples)),
        )
        logging.info("cached_features_file = %s" % cached_features_file)
        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not no_cache) or (mode == "dev" and args.use_cached_eval_features)
        ):
            features = torch.load(cached_features_file)
            logger.info(f" Features loaded from cache at {cached_features_file}")
        else:
            logger.info(" Converting to features started.")

            features = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=not evaluate,
                tqdm_enabled=not args.silent,
                threads=args.process_count,
                args=args,
            )
            if not no_cache:
                torch.save(features, cached_features_file)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

        if evaluate:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask
            )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_is_impossible,
            )

        if output_examples:
            return dataset, examples, features
        return dataset

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def _get_inputs_dict(self, batch):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }

        if self.args.model_type in ["xlm", "roberta", "distilbert", "camembert", "electra", "xlmroberta", "bart"]:
            del inputs["token_type_ids"]

        if self.args.model_type in ["xlnet", "xlm"]:
            inputs.update({"cls_index": batch[5], "p_mask": batch[6]})

        return inputs

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
        args = QuestionAnsweringArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]
