import os
import sys
import json
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
from Retrieval.cross_encoder import CrossDenseRetrieval
from Retrieval.hybrid_retriever import HybridRetriever
from dataclasses import dataclass, field
import nltk
import wandb
import yaml
from datetime import datetime
import pytz

def set_all_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# secrets.yaml 읽기 전용
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

@dataclass
class ModuleArguments:
    do_mrc: bool = field(default=False, metadata={"help": "Whether to train/evaluate/predict MRC model"})
    do_retrieval: bool = field(default=False, metadata={"help": "Whether to train/evaluate/predict retrieval model"})
    do_both: bool = field(default=False, metadata={"help": "Whether to train/evaluate/predict both models"})

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

def set_hyperparameters(config, training_args):
    training_args.num_train_epochs = config.training.epochs()
    training_args.per_device_train_batch_size = config.training.batch_size()
    training_args.per_device_eval_batch_size = config.training.batch_size()
    training_args.learning_rate = float(config.training.learning_rate())
    training_args.weight_decay = float(config.training.weight_decay())
    training_args.lr_scheduler_type  = config.training.scheduler()
    training_args.predict_with_generate  = config.training.predict_with_generate()
    training_args.save_strategy = 'epoch'
    training_args.evaluation_strategy = 'epoch'
    training_args.save_total_limit = 2
    training_args.logging_strategy = 'steps'
    training_args.logging_steps = config.training.logging_steps(10) # Config.yaml에서 조절가능
    training_args.load_best_model_at_end = True
    training_args.remove_unused_columns = True

    return training_args

def use_dense_retrieval():
    #TODO: is_testing이 적용되도록
    config = Config(path="./dense_encoder_config.yaml")
    return CrossDenseRetrieval(config)

def use_retriever_datasets(config, tokenizer, datasets, training_args, is_testing):
    print('*****doing eval or predict*****')
    if config.dataRetrieval.type() == "sparse":
        retriever = SparseRetrieval(
            tokenize_fn=tokenizer.tokenize,
            context_path=config.dataRetrieval.context_path(),
            testing=is_testing,
        )
    elif config.dataRetrieval.type() == "dense":
        retriever = use_dense_retrieval() 
    datasets = retriever.run(datasets, training_args, config)
    return datasets

def do_mrc(config, training_args, module_args, logger, is_testing):
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

    wandb.watch(model)
    # 최소 하나의 행동(do_train, do_eval, do_predict)을 해야 함
    if not (training_args.do_train or training_args.do_eval or training_args.do_predict):
        return logger.info('Neither training, evaluation, nor prediction is enabled.')

    # Prepare tokenizer and dataset
    
    if training_args.do_predict:
        _, max_seq_length = check_no_error(config, training_args, datasets, tokenizer)
        config.dataQA.tokenizer.max_seq_length.atom = max_seq_length
    else:
        last_checkpoint, _ = check_no_error(config, training_args, datasets, tokenizer)
        column_names = datasets["train"].column_names if training_args.do_train else datasets["validation"].column_names
    
    # Generation 여부에 따라 tokenizer 선택
    if config.training.predict_with_generate():
        wrapped_tokenizer = Seq2SeqLMTokenizerWrapper(tokenizer, config)
    else:
        wrapped_tokenizer = QuestionAnsweringTokenizerWrapper(tokenizer, config)
    
    train_dataset, eval_dataset = None, None
    if training_args.do_train:
        train_dataset = datasets["train"]
        eval_dataset = datasets["validation"]
        train_dataset = train_dataset.map(
            wrapped_tokenizer.encode_train,
            batched=True,
            remove_columns=column_names,
            num_proc=config.dataQA.tokenizer.preprocess_num_workers(None),
            load_from_cache_file=not config.dataQA.overwrite_cache(False)
            )
        eval_dataset = eval_dataset.map(
            wrapped_tokenizer.encode_valid,
            batched=True,
            remove_columns=column_names,
            num_proc=config.dataQA.tokenizer.preprocess_num_workers(None),
            load_from_cache_file=not config.dataQA.overwrite_cache(False)
            )
    else:
        if module_args.do_retrieval: # retrieval도 같이 수행하도록 설정한 경우
            # 원래의 context가 아닌 retirever가 찾은 context를 dataset에 넣어서 사용한다.
            datasets = use_retriever_datasets(config, tokenizer, datasets, training_args, is_testing)
            column_names = datasets["validation"].column_names
        if training_args.do_eval:
            eval_dataset = datasets["validation"]
            eval_dataset = eval_dataset.map(
                wrapped_tokenizer.encode_valid,
                batched=True,
                remove_columns=column_names,
                num_proc=config.dataQA.tokenizer.preprocess_num_workers(None),
                load_from_cache_file=not config.dataQA.overwrite_cache(False)
                )
        elif training_args.do_predict:
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
    # Generation 여부에 따라 Data collator 선택
    if config.training.predict_with_generate():
        data_collator = DataCollatorForSeq2Seq(
                wrapped_tokenizer.tokenizer,
                model=model,
                pad_to_multiple_of=8 if training_args.fp16 else None
            )
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

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
            eval_examples=datasets["validation"] if training_args.do_eval or training_args.do_train else None, # train때도 필요
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
        wrapped_tokenizer=wrapped_tokenizer
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

def do_retrieval(config, training_args, logger, is_testing):
    if config.dataRetrieval.type() == "dense":
        retriever = use_dense_retrieval()
    elif config.dataRetrieval.type() == "sparse":
        retriever = SparseRetrieval(
            tokenize_fn=AutoTokenizer.from_pretrained(config.model.name()).tokenize,
            context_path=config.dataRetrieval.context_path(),
            testing=is_testing,
        )
    elif config.dataRetrieval.type() == "hybrid":
        dense_retriever = use_dense_retrieval()
        sparse_retriever = SparseRetrieval(
            tokenize_fn=AutoTokenizer.from_pretrained(config.model.name()).tokenize,
            context_path=config.dataRetrieval.context_path(),
            testing=is_testing,
        )
        retriever = HybridRetriever(dense_retriever, sparse_retriever)
    
    if training_args.do_train:
        retriever.train()
    
    elif training_args.do_eval:
        datasets = load_from_disk(config.dataQA.path.train('./data/train_dataset'))
        if is_testing:
            datasets = use_small_datasets(datasets)
        validation_dataset = datasets["validation"]
        # TODO: faiss에 대한 설정 필요
        k = config.dataRetrieval.top_k(5)
        df = retriever.retrieve(validation_dataset, 
                                topk=k,
                                concat_context=False)
        rankings = []
        for _, row in df.iterrows():
            in_k = False
            for rank, context in enumerate(row["context"]):
                if row["original_context"] == context:
                    in_k = True
                    rankings.append(rank+1)
                    break
            if not in_k:
                rankings.append(k+1)

        def recall_at_k(k):
            return sum([1 for rank in rankings if rank <= k]) / len(rankings)
        
        recalls = {f"recall@{i}": recall_at_k(i) for i in range(1, k+1)}
        output_path = config.output.train('./outputs/train_dataset')
        with open(os.path.join(output_path, 'recalls.json'), 'w') as f:
            json.dump(recalls, f, indent=4)
            
        print(f"Recall@{k} : {recalls[f'recall@{k}']}")
    
    elif training_args.do_predict:
        datasets = load_from_disk(config.dataQA.path.test('./data/test_dataset'))
        if is_testing:
            datasets = use_small_datasets(datasets)
        datasets = retriever.run(datasets, training_args, config)
        validation_dataset = datasets["validation"]
        
        contexts = {row["id"]: row["context"] for row in validation_dataset}
        output_path = config.output.test('./outputs/test_dataset')
        with open(os.path.join(output_path, "contexts.json"), 'w') as f:
            json.dump(contexts, f, indent=4)
        print("Retrieval results are saved in", config.output.test('./outputs/test_dataset'))
        
def main():
    config = Config()
    logger = configure_logging()
    
    nltk.download('punkt')
    nltk.download('punkt_tab')

    secrets = load_config(config_path="secrets.yaml")
    wandb.login(key=secrets["wandb-api-key"])
    
    model_name = config.model.name
    tz = pytz.timezone('Asia/Seoul')
    start_time = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{start_time}"

    wandb.init(project="MRC_project", config=config, name=run_name)
    
    # Start wandb monitoring the model and parameters
    
    # Argument parsing
    parser = HfArgumentParser((CustomTrainingArguments, DataArguments, ModuleArguments))
    training_args, data_args, module_args = parser.parse_args_into_dataclasses()
    training_args.output_dir = use_proper_output_dir(config, training_args)
    training_args = set_hyperparameters(config, training_args)
    is_testing = True if data_args.testing else config.testing(False)
    set_all_seed(config.seed())

    if not (module_args.do_mrc or module_args.do_retrieval or module_args.do_both): 
        module_args.do_mrc = True
    
    if training_args.do_predict:
        module_args.do_retrieval = True # test dataset에는 context가 존재하지 않는다. 강제로 retrieval을 수행하도록 함
    
    if module_args.do_both:
        module_args.do_mrc = True
        module_args.do_retrieval = True

    if training_args.do_train and (module_args.do_mrc and module_args.do_retrieval):
        # MRC와 retrieval를 동시에 학습하는 것을 지양한다.
        # MRC와 retrieval을 각각 학습하도록 요청한다.
        return logger.info('Both MRC and retrieval training is not allowed. Please train them separately.')
    
    logger.info("Training/evaluation parameters %s", training_args)

    # 상황에 맞는 모델 이름과 데이터 경로 설정
    print('output_dir', training_args.output_dir)

    if module_args.do_mrc:
        # mrc만 수행하거나 mrc와 retrieval을 동시에 수행하는 경우
        do_mrc(config, training_args, module_args, logger, is_testing)
    elif module_args.do_retrieval:
        # retrieval만 수행하는 경우
        do_retrieval(config, training_args, logger, is_testing)

if __name__ == "__main__":
    main()