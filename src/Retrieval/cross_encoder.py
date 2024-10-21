import json
import sys
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
import random
import torch
import hashlib
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datasets import Dataset, DatasetDict, Features, Sequence, Value, concatenate_datasets, load_from_disk
from tqdm.auto import tqdm
from tqdm import trange
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, BertModel, BertPreTrainedModel, AdamW, get_linear_schedule_with_warmup
)
from NegativeSampler import NegativeSampler
from SparseNegativeSampler import SparseNegativeSampler
from sparse_retrieval import SparseRetrieval

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def dummy_row(question):
    hashed = hashlib.sha256()
    hashed.update(question.encode())
    return {
        "question": [question],
        "context": [""],
        "id": [hashed.hexdigest()]
    }
    
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class CrossDenseRetrieval:
    def __init__(self) -> NoReturn:

        self.config = Config(path='./dense_encoder_config.yaml')

        set_seed(self.config.seed())

        data_path = os.path.dirname(self.config.dataset.train_path())
        context_path = self.config.dataset.context_path()

        self.data_path = data_path
        with open(context_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = load_from_disk("./data/train_dataset/")['train']
        self.max_len = 512
        
        self.model_name = self.config.model.name()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1).to(self.device)
        self.sampler = SparseNegativeSampler(self.contexts)
        sparse_retriever = SparseRetrieval(self.tokenizer.tokenize)
        self.sampler.make_sparse_embedding(sparse_retriever)
        
        self.num_negatives = self.config.training.num_negative()
        self.args =TrainingArguments(
            output_dir=self.config.training.output_dir(),
            learning_rate=float(self.config.training.learning_rate()),
            per_device_train_batch_size=self.config.training.per_device_train_batch_size(),
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size(),
            num_train_epochs=self.config.training.epochs(),
            weight_decay=self.config.training.weight_decay(),
            )
        self.indexer = None

    def prepare_in_batch_negative(self, dataset=None, num_neg=None):
        if dataset is None:
            dataset = self.dataset
        if num_neg is None:
            num_neg = self.num_negatives

        q_with_neg = self.sampler.offer_bulk(dataset, num_neg)

        questions = q_with_neg["question"]
        contexts = q_with_neg["context"]
        negatives = q_with_neg["negatives"]

        full_questions = [Q for Q in questions for _ in range(num_neg + 1)]
        tmp =[[C] + Ns.tolist() for (C, Ns) in zip(contexts, negatives)]
        full_contexts = []
        for a in tmp:
            full_contexts += a
        
        tokenized = self.tokenizer(
            full_questions,
            full_contexts,
            padding="max_length",
            truncation="only_second",
            max_length=self.max_len
        )
        
        # labels = ([1] + [0] * num_neg) * len(questions)
                    
        input_ids = torch.tensor(tokenized["input_ids"]).view(-1, num_neg + 1, self.max_len)
        attention_mask = torch.tensor(tokenized["attention_mask"]).view(-1, num_neg + 1, self.max_len)
        token_type_ids = torch.tensor(tokenized["token_type_ids"]).view(-1, num_neg + 1, self.max_len)
        # labels = torch.tensor(labels)
        train_dataset = TensorDataset(input_ids, attention_mask, token_type_ids)

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.per_device_train_batch_size, shuffle=True)
    
    def train(self, args=None):
        if args is None:
            args = self.args

        print("Training encoder")

        batch_size = args.per_device_train_batch_size

        self.prepare_in_batch_negative()

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total
        )

        global_step = 0

        self.model.zero_grad()
        torch.cuda.empty_cache()

        for i in tqdm(range(int(args.num_train_epochs)), desc="Epoch"):
            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    self.model.train()

                    targets = torch.zeros(batch_size).long().to(self.device)
                    inputs = {
                        "input_ids": batch[0].view(-1,self.max_len).to(self.device),
                        "attention_mask": batch[1].view(-1,self.max_len).to(self.device),
                        "token_type_ids": batch[2].view(-1,self.max_len).to(self.device)
                    }

                    outputs = self.model(**inputs)

                    outputs = outputs.logits.view(batch_size, self.num_negatives + 1)

                    # 여기부터
                    sim_scores = F.log_softmax(outputs, dim=1)
                
                    loss = F.nll_loss(sim_scores, targets)
                    
                    tepoch.set_postfix(loss=f"{str(loss.item())}", step=f"{global_step+1}/{t_total}", lr=f"{scheduler.get_last_lr()[0]:.8f}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.model.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()
                    del inputs
                    
        self.model.save_pretrained(os.path.join(self.data_path, f"cross_encoder"))

    def retrieve(self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1, do_predict: bool = False) -> Union[Tuple[List, List], pd.DataFrame]: 
        
        # contexts_candidate -> 정답이 될 수 있는 passage 후보군
        # 따라서 이 retriever의 고점은 sampler의 성능을 넘길 수 없다.
        if isinstance(query_or_dataset, str):
            contexts_candidate = self.sampler.offer(dummy_row(query_or_dataset), 
                                                    25, exclude_positive=False)["negatives"]
            doc_scores, doc_indices = self.get_relevant_doc(
                query_or_dataset, contexts_candidate, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(contexts_candidate[doc_indices[i]])

            return (doc_scores, [contexts_candidate[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            contexts_candidate = self.sampler.offer_bulk(query_or_dataset, 25, exclude_positive=False, do_predict=do_predict)["negatives"]
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], contexts_candidate, k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
            ):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join(
                        [contexts_candidate[idx][pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas
        
    def get_relevant_doc(self, query: str, contexts: List[str], k: Optional[int] = 1) -> Tuple[List, List]:
        
        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        self.model.eval()
        self.model.to(self.device)
        questions = [query] * len(contexts)

        with torch.no_grad():
            tokenized = self.tokenizer(
                questions,
                contexts.tolist(),
                padding="max_length",
                truncation="only_second",
                max_length=self.max_len
            )
            tokenized = {key: torch.tensor(val).to(self.device) for key, val in tokenized.items()}
            scores = self.model(**tokenized).logits.cpu()
            scores = scores.view(-1)
            doc_score, doc_indices = torch.topk(scores, k=k)
        torch.cuda.empty_cache()

        return doc_score.tolist(), doc_indices.tolist()

    def get_relevant_doc_bulk(self, queries: List[str], contexts_list: List[List[str]],  k: Optional[int] = 1) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Arguments:
            queries (List[str]):
                여러 개의 Query를 리스트로 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Returns:
            Tuple[List[List[float]], List[List[int]]]:
                각 쿼리에 대한 상위 k개 문서의 점수와 인덱스를 반환합니다.
        """

        self.model.eval()
        self.model.to(self.device)

        batch_size = 16  # Adjust this value based on your memory constraints
        all_scores = []

        with torch.no_grad():
            num_samples = len(contexts_list[0])
            for i in tqdm(range(0, len(queries), batch_size), desc="Dense retrieval: "):
                batch_queries = queries[i:i+batch_size]
                batch_contexts = contexts_list[i:i+batch_size]
                
                full_questions = [Q for Q in batch_queries for _ in range(num_samples)]
                full_contexts = []
                for contexts in batch_contexts:
                    full_contexts += contexts.tolist()
                
                tokenized = self.tokenizer(
                    full_questions,
                    full_contexts,
                    padding="max_length",
                    truncation="only_second",
                    max_length=self.max_len
                )
                tokenized = {key: torch.tensor(val).to(self.device) for key, val in tokenized.items()}
                scores = self.model(**tokenized).logits.cpu().view(-1, num_samples)
                all_scores.append(scores)

            all_scores = torch.cat(all_scores, dim=0)
            doc_score, doc_indices = torch.topk(all_scores, k=k, dim=1)

        torch.cuda.empty_cache()

        return doc_score.tolist(), doc_indices.tolist()

    def run(self, datasets, training_args, config):

        if os.path.exists(os.path.join(self.data_path, "cross_encoder")):
            self.model = AutoModelForSequenceClassification.from_pretrained(os.path.join(self.data_path, "cross_encoder"))
        else:
            self.train()
    
        
        df = self.retrieve(
            datasets["validation"],
            topk=config.dataRetreival.top_k(5),
            do_predict = training_args.do_predict
        )
        
        if training_args.do_predict:
            f = Features(
                {
                    "context": Value(dtype="string", id=None),
                    "id": Value(dtype="string", id=None),
                    "question": Value(dtype="string", id=None),
                }
            )
        elif training_args.do_eval:
            f = Features(
                {
                    "answers": Sequence(
                        feature={
                            "text": Value(dtype="string", id=None),
                            "answer_start": Value(dtype="int32", id=None),
                        },
                        length=-1,
                        id=None,
                    ),
                    "context": Value(dtype="string", id=None),
                }
            )
        
        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        return datasets

if __name__ == "__main__":

    retriever = CrossDenseRetrieval()

    # Test sparse
    org_dataset = load_from_disk(os.path.join(retriever.data_path, "./train_dataset"))
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)


    if os.path.exists(os.path.join(retriever.data_path, "cross_encoder")):
        retriever.model = AutoModelForSequenceClassification.from_pretrained(os.path.join(retriever.data_path, "cross_encoder"))
    else:
        retriever.train()
    
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    with timer("bulk query by exhaustive search"):
        df = retriever.retrieve(full_ds, topk=1)
        df["correct"] = df["correct"] = df.apply(lambda row: row["original_context"] in row["context"], axis=1)
        print(
            "correct retrieval result by exhaustive search",
            df["correct"].sum() / len(df),
        )

    with timer("single query by exhaustive search"):
        scores, indices = retriever.retrieve(query, topk=1)