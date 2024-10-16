import os
import sys
import random
import logging
import argparse
import numpy as np
import torch
import nltk
# from datasets import load_from_disk, load_metric 
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
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from config import Config
from QuestionAnswering.tokenizer_wrapper import QuestionAnsweringTokenizerWrapper, Seq2SeqLMTokenizerWrapper
from QuestionAnswering.trainer import QuestionAnsweringTrainer, GenerationBasedSeq2SeqTrainer
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

def set_hyperparameters(config, training_args):
    training_args.num_train_epochs = config.training.epochs()
    training_args.per_device_train_batch_size = config.training.batch_size()
    training_args.per_device_eval_batch_size = config.training.batch_size()
    training_args.learning_rate = float(config.training.learning_rate())
    training_args.weight_decay = float(config.training.weight_decay())
    training_args.lr_scheduler_type  = config.training.scheduler()
    training_args.save_strategy = 'epoch',
    training_args.evaluation_strategy = 'epoch',
    training_args.save_total_limit = 2,
    training_args.logging_strategy = 'epoch',
    training_args.load_best_model_at_end = True,
    training_args.remove_unused_columns = True

    return training_args

def main():
    config = Config()
    
    set_all_seed(config.seed())
    logger = logging.getLogger(__name__)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--testing", action='store_true', help="Use only 1% of the dataset for testing")
    args, unknown = arg_parser.parse_known_args()    
    
    parser = HfArgumentParser((TrainingArguments,))
    training_args = parser.parse_args_into_dataclasses(unknown)[0]
    training_args = set_hyperparameters(config, training_args)

    training_args.predict_with_generate = True
    
    logger.info("Training/evaluation parameters %s", training_args)
    
    datasets = load_from_disk(config.dataQA.path())
    
    if args.testing:
        # 1%만 선택
        num_train_samples = len(datasets['train'])
        num_valid_samples = len(datasets["validation"])
        test_samples = int(0.01 * num_train_samples)
        valid_samples = int(0.01 * num_valid_samples)
        
        datasets["train"] = datasets["train"].select(range(test_samples))
        datasets["validation"] = datasets["validation"].select(range(valid_samples))
    
    
    print(datasets)
    
    config_hf = AutoConfig.from_pretrained(config.model.name())
    tokenizer = AutoTokenizer.from_pretrained(config.model.name(), use_fast=True)
    # model = AutoModelForQuestionAnswering.from_pretrained(config.model.name(), config=config_hf)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model.name(), config=config_hf)
    
    last_checkpoint, _ = check_no_error(config, training_args, datasets, tokenizer) 
    
    if not (training_args.do_train or training_args.do_eval):
        return
        
    # wrapped_tokenizer = QuestionAnsweringTokenizerWrapper(tokenizer, config)
    wrapped_tokenizer = Seq2SeqLMTokenizerWrapper(tokenizer, config)
    column_names = datasets["train"].column_names if training_args.do_train else datasets["validation"].column_names

    print(f'do_train = {training_args.do_train}')
    print(f'do_eval = {training_args.do_eval}')

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
    
    # data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    metric = load_metric("squad")
    # def compute_metrics(eval_predictions):
    #     return metric.compute(predictions=eval_predictions.predictions,
    #                             references=eval_predictions.label_ids)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        # preds에서 토큰 ID 리스트 추출
        token_ids = [pred["prediction_text"] for pred in preds]
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(preds)
        decoded_preds = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        # decoded_labels is for rouge metric, not used for f1/em metric

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        formatted_predictions = [{"id": ex['id'], "prediction_text": decoded_preds[i]} for i, ex in enumerate(datasets["validation"])]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]

        result = metric.compute(predictions=formatted_predictions, references=references)
        return result
    
    # trainer = QuestionAnsweringTrainer(
    #     model=model,
    #     args=training_args,
    #     config=config,
    #     train_dataset=train_dataset if training_args.do_train else None,
    #     eval_dataset=eval_dataset if training_args.do_eval else None,
    #     eval_examples=datasets["validation"] if training_args.do_eval else None,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    #     wrapped_tokenizer=wrapped_tokenizer,
    # )
    trainer = GenerationBasedSeq2SeqTrainer(
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