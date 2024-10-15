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
from transformers.trainer_utils import get_last_checkpoint

from config import Config
from QuestionAnswering.tokenizer_wrapper import QuestionAnsweringTokenizerWrapper
from QuestionAnswering.trainer import QuestionAnsweringTrainer
from QuestionAnswering.utils import check_no_error


def set_all_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_logging():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logger


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--testing", action='store_true', help="Use only 1% of the dataset for testing")
    args, unknown = arg_parser.parse_known_args()

    parser = HfArgumentParser((TrainingArguments,))
    training_args = parser.parse_args_into_dataclasses(unknown)[0]
    
    return args, training_args


def load_datasets(config, is_testing=False):
    datasets = load_from_disk(config.dataQA.path())
    if is_testing:
        num_train_samples = len(datasets['train'])
        num_valid_samples = len(datasets["validation"])
        datasets["train"] = datasets["train"].select(range(int(0.01 * num_train_samples)))
        datasets["validation"] = datasets["validation"].select(range(int(0.01 * num_valid_samples)))
    return datasets


def set_hyperparameters(config, training_args):
    training_args.num_train_epochs = config.training.epochs()
    training_args.per_device_train_batch_size = config.training.batch_size()
    training_args.per_device_eval_batch_size = config.training.batch_size()
    training_args.learning_rate = float(config.training.learning_rate())
    training_args.weight_decay = float(config.training.weight_decay())
    training_args.lr_scheduler_type = config.training.scheduler()
    return training_args


def prepare_tokenizer_and_model(config):
    config_hf = AutoConfig.from_pretrained(config.model.name())
    tokenizer = AutoTokenizer.from_pretrained(config.model.name(), use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(config.model.name(), config=config_hf)
    return tokenizer, model


def process_datasets(datasets, wrapped_tokenizer, column_names, is_train=True, config=None):
    dataset_type = "train" if is_train else "validation"
    dataset = datasets[dataset_type]
    dataset = dataset.map(
        wrapped_tokenizer.encode_train if is_train else wrapped_tokenizer.encode_valid,
        batched=True,
        remove_columns=column_names,
        num_proc=config.dataQA.tokenizer.preprocess_num_workers(None),
        load_from_cache_file=not config.dataQA.overwrite_cache(False)
    )
    return dataset


def main():
    config = Config()
    logger = configure_logging()
    args, training_args = parse_arguments()
    
    set_all_seed(config.seed())
    training_args = set_hyperparameters(config, training_args)

    logger.info("Training/evaluation parameters %s", training_args)
    
    datasets = load_datasets(config, args.testing)
    print(datasets)
    
    tokenizer, model = prepare_tokenizer_and_model(config)
    
    last_checkpoint, _ = check_no_error(config, training_args, datasets, tokenizer)
    
    if not (training_args.do_train or training_args.do_eval):
        return print('there is no command --do_train or --do_eval')
        
    wrapped_tokenizer = QuestionAnsweringTokenizerWrapper(tokenizer, config)
    column_names = datasets["train"].column_names if training_args.do_train else datasets["validation"].column_names
    
    train_dataset, eval_dataset = None, None
    if training_args.do_train:
        train_dataset = process_datasets(datasets, wrapped_tokenizer, column_names, is_train=True, config=config)

    if training_args.do_eval:
        eval_dataset = process_datasets(datasets, wrapped_tokenizer, column_names, is_train=False, config=config)

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    
    metric = load_metric("squad")
    
    def compute_metrics(eval_predictions):
        return metric.compute(predictions=eval_predictions.predictions, references=eval_predictions.label_ids)

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        wrapped_tokenizer=wrapped_tokenizer,
    )
    
    if training_args.do_train:
        checkpoint = last_checkpoint or (config.model.name() if os.path.isdir(config.model.name()) else None)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")
        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
    
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
