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

from config import Config
from QuestionAnswering.utils import check_no_error
from QuestionAnswering.trainer import QuestionAnsweringTrainer
from QuestionAnswering.tokenizer_wrapper import QuestionAnsweringTokenizerWrapper
from Retrieval.sparse_retrieval import SparseRetrieval

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
    training_args.do_train = True
    
    logger.info("Training/evaluation parameters %s", training_args)
    
    datasets = load_from_disk(config.dataQA.path())
    print(datasets)
    
    config_hf = AutoConfig.from_pretrained(config.model.name())
    tokenizer = AutoTokenizer.from_pretrained(config.model.name(), use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(config.model.name(), config=config_hf)
    
    if config.dataRetrieval.eval(True):
        retriever = SparseRetrieval(
            tokenize_fn=tokenizer.tokenize,
            context_path=config.dataRetrieval.context_path(),
        )
        datasets = retriever.run(datasets, training_args, config)

    if not (training_args.do_eval or training_args.do_predict):
        return

    _, max_seq_length = check_no_error(config, training_args, datasets, tokenizer)
    config.dataQA.tokenizer.max_seq_length.atom = max_seq_length
    wrapped_tokenizer = QuestionAnsweringTokenizerWrapper(tokenizer, config)
    
    eval_dataset = datasets["validation"]
    eval_dataset = eval_dataset.map(
        wrapped_tokenizer.encode_valid,
        batched=True,
        remove_columns=eval_dataset.column_names,
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
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        wrapped_tokenizer=wrapped_tokenizer
    )
    
    logger.info("*** Evaluate ***")

    #### eval dataset & eval example - predictions.json 생성됨
    if training_args.do_predict:
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )

        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
    
if __name__ == "__main__":
    main()