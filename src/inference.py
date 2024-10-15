import os
import sys
import random
import logging
import argparse
import numpy as np
import torch
from datasets import load_from_disk
from evaluate import load as load_metric
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from config import Config
from QuestionAnswering.utils import check_no_error
from QuestionAnswering.trainer import QuestionAnsweringTrainer
from QuestionAnswering.tokenizer_wrapper import QuestionAnsweringTokenizerWrapper
from Retrieval.sparse_retrieval import SparseRetrieval
from dataclasses import dataclass, field

def set_all_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@dataclass
class DataArguments:
    dataset_name: str = field(default=None, metadata={"help": "The name of the dataset to use."})
    testing: bool = field(default=False, metadata={"help": "Use only 1% of the dataset for testing"})

def configure_logging():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logger

def load_and_process_datasets(config, data_args):
    if data_args.dataset_name:
        datasets = load_from_disk(data_args.dataset_name)
    else:
        datasets = load_from_disk(config.dataQA.path())

    if data_args.testing:
        num_valid_samples = len(datasets["validation"])
        datasets["validation"] = datasets["validation"].select(range(int(0.01 * num_valid_samples)))
    
    return datasets

def main():
    config = Config()
    logger = configure_logging()
    
    # Argument parsing
    parser = HfArgumentParser((TrainingArguments, DataArguments))
    training_args, data_args = parser.parse_args_into_dataclasses()
    training_args.do_train = False  # inference에서는 학습을 하지 않음
    
    set_all_seed(config.seed())

    logger.info("Training/evaluation parameters %s", training_args)
    
    # Load datasets
    datasets = load_and_process_datasets(config, data_args)
    print(datasets)

    # Load model and tokenizer
    config_hf = AutoConfig.from_pretrained(config.model.name())
    tokenizer = AutoTokenizer.from_pretrained(config.model.name(), use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(config.model.name(), config=config_hf)
    
    # Sparse Retrieval
    if config.dataRetrieval.eval(True):
        retriever = SparseRetrieval(
            tokenize_fn=tokenizer.tokenize,
            context_path=config.dataRetrieval.context_path(),
            testing=data_args.testing
        )
        datasets = retriever.run(datasets, training_args, config)
    
    # Ensure at least one action (evaluation or prediction) is specified
    if not (training_args.do_eval or training_args.do_predict):
        return logger.info('Neither evaluation nor prediction is enabled.')

    # Prepare tokenizer and dataset
    _, max_seq_length = check_no_error(config, training_args, datasets, tokenizer)
    config.dataQA.tokenizer.max_seq_length.atom = max_seq_length
    wrapped_tokenizer = QuestionAnsweringTokenizerWrapper(tokenizer, config)
    
    eval_dataset = datasets["validation"]
    eval_dataset = eval_dataset.map(
        wrapped_tokenizer.encode_valid,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # Metric for evaluation
    metric = load_metric("squad")
    def compute_metrics(eval_predictions):
        return metric.compute(predictions=eval_predictions.predictions, references=eval_predictions.label_ids)
    
    # Trainer for evaluation and prediction
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        config=config,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        wrapped_tokenizer=wrapped_tokenizer,
    )

    # Prediction
    if training_args.do_predict:
        logger.info("*** Prediction ***")
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )
        logger.info("No metric can be presented because there is no correct answer given. Job done!")

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

if __name__ == "__main__":
    main()
