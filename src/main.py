import os
import sys
import random
import logging
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
    testing: bool = field(default=False, metadata={"help": "Use only 1% of the dataset for testing"})

@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str =field(default="./outputs", metadata = {"help": "The output directory"})

def configure_logging():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logger

def use_proper_datasets(config, training_args):
    if training_args.do_train or training_args.do_eval:
        return load_from_disk(config.dataQA.path.train('./data/train_dataset')) 
    elif training_args.do_predict:
        return load_from_disk(config.dataQA.path.test('./data/test_dataset'))
    else:
        return None

def use_small_datasets(datasets):
    def cut(dataset):
        length = len(dataset)
        return dataset.select(range(length // 100))
    return {key: cut(dataset) for key, dataset in datasets.items()}

def use_proper_output_dir(config, training_args):
    if training_args.do_train:
        # do_train의 결과는 model이므로
        return config.output.model('./models/train_dataset')
    elif training_args.do_eval:
        # do_eval의 결과는 evaluation 결과이므로
        return config.output.train('./outputs/train_dataset')
    elif training_args.do_predict:
        # do_predict의 결과는 prediction 결과이므로
        return config.output.test('./outputs/test_dataset')
    else:
        return None

def use_proper_model(config, training_args):
    if training_args.do_train:
        return config.model.name()
    elif training_args.do_eval or training_args.do_predict:
        return config.output.model('./models/train_dataset')

def main():
    config = Config()
    logger = configure_logging()
    
    # Argument parsing
    parser = HfArgumentParser((CustomTrainingArguments, DataArguments))
    training_args, data_args = parser.parse_args_into_dataclasses()
    training_args.output_dir = use_proper_output_dir(config, training_args)
    is_testing = True if data_args.testing else config.testing(False)
    set_all_seed(config.seed())

    logger.info("Training/evaluation parameters %s", training_args)

    # 상황에 맞는 모델 이름과 데이터 경로 설정
    print('output_dir', training_args.output_dir)

    datasets = use_proper_datasets(config, training_args)
    if is_testing:
        datasets = use_small_datasets(datasets)

    # Load model and tokenizer
    
    model_name = use_proper_model(config, training_args)
    config_hf = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config_hf)
    
    # Sparse Retrieval
    if config.dataRetrieval.eval(True) and training_args.do_predict:
        print('*****doing eval or predict*****')
        retriever = SparseRetrieval(
            tokenize_fn=tokenizer.tokenize,
            context_path=config.dataRetrieval.context_path(),
            testing=is_testing,
        )
        datasets = retriever.run(datasets, training_args, config)

    # 최소 하나의 행동(do_train, do_eval, do_predict)을 해야 함
    if not (training_args.do_train or training_args.do_eval or training_args.do_predict):
        return logger.info('Neither training, evaluation, nor prediction is enabled.')

    # Prepare tokenizer and dataset
    
    if training_args.do_predict:
        _, max_seq_length = check_no_error(config, training_args, datasets, tokenizer)
        config.dataQA.tokenizer.max_seq_length.atom = max_seq_length
        wrapped_tokenizer = QuestionAnsweringTokenizerWrapper(tokenizer, config)
        
    else:
        last_checkpoint, _ = check_no_error(config, training_args, datasets, tokenizer)
        wrapped_tokenizer = QuestionAnsweringTokenizerWrapper(tokenizer, config)
        column_names = datasets["train"].column_names if training_args.do_train else datasets["validation"].column_names
    
    train_dataset, eval_dataset = None, None
    if training_args.do_train:
        train_dataset = datasets["train"]
        train_dataset = train_dataset.map(
            wrapped_tokenizer.encode_train,
            batched=True,
            remove_columns=column_names,
            num_proc=config.dataQA.tokenizer.preprocess_num_workers(None),
            load_from_cache_file=not config.dataQA.overwrite_cache(False)
            )
    elif training_args.do_eval:
        eval_dataset = datasets["validation"]
        eval_dataset = eval_dataset.map(
            wrapped_tokenizer.encode_valid,
            batched=True,
            remove_columns=column_names,
            num_proc=config.dataQA.tokenizer.preprocess_num_workers(None),
            load_from_cache_file=not config.dataQA.overwrite_cache(False)
            )
    else:
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
    
    # Trainer for training, evaluation, and prediction
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
    
    # Training
    if training_args.do_train:
        logger.info("*** Training ***")
        checkpoint = last_checkpoint or (config.model.name() if os.path.isdir(config.model.name()) else None)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Prediction
    if training_args.do_predict:
        logger.info("*** Prediction ***")
        predictions = trainer.predict(test_dataset=eval_dataset, test_examples=datasets["validation"])
        logger.info("No metric can be presented because there is no correct answer given. Job done!")
        
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
