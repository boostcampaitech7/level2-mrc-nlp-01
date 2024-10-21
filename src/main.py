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
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from config import Config
from QuestionAnswering.utils import check_no_error
from QuestionAnswering.trainer import QuestionAnsweringTrainer, GenerationBasedSeq2SeqTrainer
from QuestionAnswering.tokenizer_wrapper import QuestionAnsweringTokenizerWrapper, Seq2SeqLMTokenizerWrapper
from Retrieval.sparse_retrieval import SparseRetrieval
from Retrieval.dense_retrieval import DenseRetrieval
from dataclasses import dataclass, field
import nltk

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
class DataArguments:
    testing: bool = field(default=False, metadata={"help": "Use only 1% of the dataset for testing"})

@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str =field(default="./outputs", metadata = {"help": "The output directory"})
    reader_only: bool = field(default=False, metadata={"help": "Whether to use reader only"})

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

def set_hyperparameters(config, training_args):
    training_args.num_train_epochs = config.training.epochs()
    training_args.per_device_train_batch_size = config.training.batch_size()
    training_args.per_device_eval_batch_size = config.training.batch_size()
    training_args.learning_rate = float(config.training.learning_rate())
    training_args.weight_decay = float(config.training.weight_decay())
    training_args.lr_scheduler_type  = config.training.scheduler()
    training_args.predict_with_generate  = config.training.predict_with_generate()
    training_args.save_strategy = 'epoch',
    training_args.evaluation_strategy = 'epoch',
    training_args.save_total_limit = 2,
    training_args.logging_strategy = 'epoch',
    training_args.load_best_model_at_end = True,
    training_args.remove_unused_columns = True

    return training_args

def main():
    config = Config()
    logger = configure_logging()
    
    nltk.download('punkt')
    nltk.download('punkt_tab')
    
    # Argument parsing
    parser = HfArgumentParser((CustomTrainingArguments, DataArguments))
    training_args, data_args = parser.parse_args_into_dataclasses()
    training_args.output_dir = use_proper_output_dir(config, training_args)
    training_args = set_hyperparameters(config, training_args)
    is_testing = True if data_args.testing else config.testing(False)
    training_args.output_dir = use_proper_output_dir(config, training_args)
    training_args = set_hyperparameters(config, training_args)
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
    # Generation 여부에 따라 모델 선택
    if config.training.predict_with_generate():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config_hf)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config_hf)
    
    # Sparse Retrieval
    if config.dataRetrieval.eval(True) and (training_args.do_predict or training_args.do_eval) and not training_args.reader_only:
        print('*****doing eval or predict*****')
        if config.dataRetrieval.type() == "sparse":
            retriever = SparseRetrieval(
                tokenize_fn=tokenizer.tokenize,
                context_path=config.dataRetrieval.context_path(),
                testing=is_testing,
            )
        elif config.dataRetrieval.type() == "dense":
            retriever = DenseRetrieval()
        datasets = retriever.run(datasets, training_args, config)

    # 최소 하나의 행동(do_train, do_eval, do_predict)을 해야 함
    if not (training_args.do_train or training_args.do_eval or training_args.do_predict):
        return logger.info('Neither training, evaluation, nor prediction is enabled.')

    # Prepare tokenizer and dataset
    
    if training_args.do_predict:
        _, max_seq_length = check_no_error(config, training_args, datasets, tokenizer)
        config.dataQA.tokenizer.max_seq_length.atom = max_seq_length
        # Generation 여부에 따라 tokenizer 선택
        if config.training.predict_with_generate():
            wrapped_tokenizer = Seq2SeqLMTokenizerWrapper(tokenizer, config)
        else:
            wrapped_tokenizer = QuestionAnsweringTokenizerWrapper(tokenizer, config)
        
    else:
        last_checkpoint, _ = check_no_error(config, training_args, datasets, tokenizer)
        # Generation 여부에 따라 tokenizer 선택
        if config.training.predict_with_generate():
            wrapped_tokenizer = Seq2SeqLMTokenizerWrapper(tokenizer, config)
        else:
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
        # Generation 여부에 따라 encoding method 선택
        if config.training.predict_with_generate():
            eval_dataset = eval_dataset.map(
                wrapped_tokenizer.encode_test,
                batched=True,
                remove_columns=eval_dataset.column_names,
                )
        else:
            eval_dataset = eval_dataset.map(
                wrapped_tokenizer.encode_valid,
                batched=True,
                remove_columns=eval_dataset.column_names,
                )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    data_collator = DataCollatorForSeq2Seq(
            wrapped_tokenizer.tokenizer,
            model=model,
            pad_to_multiple_of=8 if training_args.fp16 else None
        )

    # Metric for evaluation
    metric = load_metric("squad")
    def compute_metrics(eval_predictions):
        return metric.compute(predictions=eval_predictions.predictions, references=eval_predictions.label_ids)
    
    # Trainer for training, evaluation, and prediction
    # Generation 여부에 따라 Trainer 선택
    if config.training.predict_with_generate():
        trainer = GenerationBasedSeq2SeqTrainer(
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
    else:
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