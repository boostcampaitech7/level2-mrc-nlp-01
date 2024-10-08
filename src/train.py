import os
import sys
import random
import logging

import numpy as np
import torch
from datasets import load_from_disk, load_metric
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

def main():
    config = Config()
    
    set_all_seed(config.seed())
    logger = logging.getLogger(__name__)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    parser = HfArgumentParser((TrainingArguments))
    training_args = parser.parse_args_into_dataclasses()[0]
    
    logger.info("Training/evaluation parameters %s", training_args)
    
    datasets = load_from_disk(config.dataQA.path())
    print(datasets)
    
    config_hf = AutoConfig.from_pretrained(config.model.name())
    tokenizer = AutoTokenizer.from_pretrained(config.model.name(), use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(config.model.name(), config=config_hf)
    
    last_checkpoint, _ = check_no_error(config, training_args, datasets, tokenizer) 
    
    if not (training_args.do_train or training_args.do_eval):
        return
        
    wrapped_tokenizer = QuestionAnsweringTokenizerWrapper(tokenizer, config)
    column_names = datasets["train"].column_names if training_args.do_train else datasets["validation"].column_names
            
    if training_args.do_train:
        train_dataset = datasets["train"]
        
        train_dataset = train_dataset.map(
            wrapped_tokenizer.encode_train,
            batched=True,
            remove_columns=column_names,
            num_proc=config.dataQA.tokenizer.preprocess_num_workers(None),
            load_from_cache_file=not config.dataQA.overwrite_cache(False)
        )
    
    if training_args.do_eval:
        eval_dataset = datasets["validation"]
        
        eval_dataset = eval_dataset.map(
            wrapped_tokenizer.encode_valid,
            batched=True,
            remove_columns=column_names,
            num_proc=config.dataQA.tokenizer.preprocess_num_workers(None),
            load_from_cache_file=not config.dataQA.overwrite_cache(False)
        )
    
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    
    metric = load_metric("squad")
    def compute_metrics(eval_predictions):
        return metric.compute(predictions=eval_predictions.predictions,
                                references=eval_predictions.label_ids)
    
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        config=config,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        wrapped_tokenizer=wrapped_tokenizer,
    )
    
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(config.model.name()):
            checkpoint = config.model.name()
        else:
            checkpoint = None
        
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

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )
    
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        
        metrics["eval_samples"] = len(eval_dataset)
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()