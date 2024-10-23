import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from tqdm.auto import tqdm
from transformers import TrainingArguments
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import logging
import torch

from .sparse_retrieval import SparseRetrieval
from .dense_retrieval import DenseRetrieval
#from .cross_encoder import CrossEncoder  # 이 import를 추가
from transformers import RobertaModel as RoBERTaEncoder, BertModel as BertEncoder



@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class Hybrid1StageRetrieval:
    def __init__(self, sparse_retriever, dense_retriever, config):
        logging.info("Initializing Hybrid1StageRetrieval")
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.ratio = config.dataRetrieval.hybrid_ratio()
        self.top_k = config.dataRetrieval.top_k()
        self.config = config  # config 저장
        print(f"Initializing hybrid_1stage with ratio: {self.ratio}")
        print(f"Initializing hybrid_1stage with top_k: {self.top_k}")
        
        # 디버그 정보 추가
        print(f"Debug: config.dataRetrieval = {config.dataRetrieval()}")
        print(f"Debug: config.dataRetrieval.hybrid_ratio = {config.dataRetrieval.hybrid_ratio()}")
        
        # get_dense_embedding() 호출 완전히 제거

    def retrieve(self, query_or_dataset, topk=None):
        print(f"Dense embeddings status: {'Loaded' if self.dense_retriever.p_embeddings is not None else 'Not loaded'}")
        if self.dense_retriever.p_embeddings is None:
            print("Getting dense embeddings...")
            if self.config.dataRetrieval.type() == "hybrid1":
                self.get_dense_embedding_hybrid1()
            else:
                self.dense_retriever.get_dense_embedding()
        
        if isinstance(query_or_dataset, Dataset):
            # Dataset인 경우의 처리
            total = []
            for example in tqdm(query_or_dataset, desc="Hybrid1 retrieval: "):
                sparse_doc_scores, sparse_doc_indices = self.sparse_retriever.retrieve(example["question"], topk=self.top_k)
                dense_doc_scores, dense_doc_indices = self.dense_retriever.retrieve(example["question"], topk=self.top_k)
                
                # Combine and re-rank
                combined_scores = []
                combined_indices = []
                for i in range(self.top_k):
                    if i < int(self.top_k * self.ratio):
                        combined_scores.append(sparse_doc_scores[i])
                        combined_indices.append(sparse_doc_indices[i])
                    else:
                        combined_scores.append(dense_doc_scores[i])
                        combined_indices.append(dense_doc_indices[i])
                
                # Re-rank based on combined scores
                sorted_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)
                reranked_indices = [combined_indices[i] for i in sorted_indices]
                
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join([self.sparse_retriever.contexts[i] for i in reranked_indices[:topk]])
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas
        
        else:
            # 단일 query 문자열인 경우의 처리 (기존 코드 유지)
            if topk is None:
                topk = self.top_k

            print(f"Retrieving with top_k: {topk}")
            print(f"Using hybrid ratio: {self.ratio}")

            if self.ratio is None:
                raise ValueError("Hybrid ratio is not set. Please check your configuration.")

            # Sparse retrieval
            print("Performing sparse retrieval...")
            sparse_results = self.sparse_retriever.retrieve(query_or_dataset, topk=topk)

            # Dense retrieval
            print("Performing dense retrieval...")
            dense_results = self.dense_retriever.retrieve(query_or_dataset, topk=topk)

            # Combine results
            print("Combining sparse and dense results...")
            combined_results = []
            for sparse_row, dense_row in tqdm(zip(sparse_results.itertuples(), dense_results.itertuples()), total=len(sparse_results), desc="Combining results"):
                sparse_contexts = sparse_row.context.split(" ")[:int(topk * (1 - self.ratio))]
                dense_contexts = dense_row.context.split(" ")[:int(topk * self.ratio)]
                
                combined_context = " ".join(sparse_contexts + dense_contexts)
                
                combined_results.append({
                    "question": sparse_row.question,
                    "id": sparse_row.id,
                    "context": combined_context,
                })

            print("Retrieval completed.")
            return pd.DataFrame(combined_results)

    def retrieve_single(self, query: str, topk: Optional[int] = 1) -> Tuple[List, List]:
        print(f"Query: {query}")
        sparse_scores, sparse_indices = self.sparse_retriever.get_relevant_doc(query, topk)
        
        try:
            dense_scores, dense_indices = self.dense_retriever.get_relevant_doc(query, topk)
        except Exception as e:
            print(f"Error in dense retriever: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {traceback.format_exc()}")
            return sparse_scores, sparse_indices

        print(f"Sparse scores: {sparse_scores}")
        print(f"Sparse indices: {sparse_indices}")
        print(f"Dense scores: {dense_scores}")
        print(f"Dense indices: {dense_indices}")
        
        merged_scores, merged_indices = self._merge_results(sparse_scores, sparse_indices, dense_scores, dense_indices, topk)
        
        print(f"Merged scores: {merged_scores}")
        print(f"Merged indices: {merged_indices}")
        
        return merged_scores, merged_indices

    def retrieve_multi(self, queries: Dataset, topk: Optional[int] = 1) -> pd.DataFrame:
        total = []
        with tqdm(total=len(queries)) as pbar:
            for i, query in enumerate(queries):
                question = query['question']
                scores, indices = self.retrieve_single(question, topk)
                
                context = [self.contexts[idx] for idx in indices]
                
                tmp = {
                    "question": question,
                    "id": query["id"],
                    "context": context,
                    "original_context": query.get("context", "")  # 원본 context 추가
                }
                if "answers" in query:
                    tmp["answers"] = query["answers"]
                
                total.append(tmp)
                pbar.update(1)

        df = pd.DataFrame(total)
        return df

    def _merge_results(self, sparse_scores, sparse_indices, dense_scores, dense_indices, topk):
        merged_dict = {}
        for score, idx in zip(sparse_scores, sparse_indices):
            merged_dict[idx] = self.ratio * score
        for score, idx in zip(dense_scores, dense_indices):
            merged_dict[idx] = merged_dict.get(idx, 0) + (1 - self.ratio) * score

        merged_items = sorted(merged_dict.items(), key=lambda x: x[1], reverse=True)[:topk]
        merged_indices, merged_scores = zip(*merged_items)
        
        return list(merged_scores), list(merged_indices)

    def run(self, datasets, training_args, config):
        print("Starting hybrid_1stage run method")
        
        print(f"Using ratio: {self.ratio}")
        print(f"Using top_k: {self.top_k}")

        # datasets가 Dataset 객체인 경우 처리
        if isinstance(datasets, Dataset):
            print("Processing single dataset")
            df = self.retrieve(datasets, topk=self.top_k)
        else:
            # datasets가 딕셔너리인 경우 (여러 split이 있는 경우)
            print("Processing multiple datasets")
            df = {}
            for split, dataset in datasets.items():
                print(f"Processing {split} split")
                df[split] = self.retrieve(dataset, topk=self.top_k)

        if training_args.do_predict:
            f = Features({
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            })
        elif training_args.do_eval:
            f = Features({
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            })

        # df가 딕셔너리인 경우 (여러 split이 있는 경우)
        if isinstance(df, dict):
            datasets = DatasetDict({split: Dataset.from_pandas(split_df, features=f) for split, split_df in df.items()})
        else:
            # df가 단일 DataFrame인 경우
            datasets = Dataset.from_pandas(df, features=f)

        return datasets

    def set_ratio(self, new_ratio: float):
        if not 0 <= new_ratio <= 1:
            raise ValueError("ratio must be between 0 and 1")
        self.ratio = new_ratio
        print(f"Ratio updated to: {self.ratio}")

    def train(self, datasets: DatasetDict, training_args: TrainingArguments, config: Dict):
        print("Starting hybrid_1stage training")
        
        # Sparse 모델 학습
        print("Training Sparse Retrieval model")
        self.sparse_retriever.train(datasets, training_args, config)
        
        # Dense 모델 학습
        print("Training Dense Retrieval model")
        try:
            dense_retriever = DenseRetrieval(config)
        except Exception as e:
            print(f"Error initializing DenseRetrieval: {e}")
            print(f"Debug: config = {config}")
            raise
        self.dense_retriever.train(datasets, training_args, config)
        
        print("Finished hybrid_1stage training")

    def get_dense_embedding_hybrid1(self):
        print("Starting get_dense_embedding_hybrid1 method")
        pickle_name = f"dense_embedding.bin"
        emd_path = os.path.join(self.dense_retriever.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.dense_retriever.p_embeddings = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            if os.path.exists(os.path.join(self.dense_retriever.data_path, "p_encoder")) and os.path.exists(os.path.join(self.dense_retriever.data_path, "q_encoder")):
                if 'roberta' in self.dense_retriever.model_name.lower() or 'roberta' in self.dense_retriever.model.config.model_type.lower():
                    self.dense_retriever.p_encoder = RoBERTaEncoder.from_pretrained(os.path.join(self.dense_retriever.data_path, "p_encoder")).to(self.dense_retriever.device)
                    self.dense_retriever.q_encoder = RoBERTaEncoder.from_pretrained(os.path.join(self.dense_retriever.data_path, "q_encoder")).to(self.dense_retriever.device)
                else:
                    self.dense_retriever.p_encoder = BertEncoder.from_pretrained(os.path.join(self.dense_retriever.data_path, "p_encoder")).to(self.dense_retriever.device)
                    self.dense_retriever.q_encoder = BertEncoder.from_pretrained(os.path.join(self.dense_retriever.data_path, "q_encoder")).to(self.dense_retriever.device)
            else:
                print("No pre-trained encoders found. Please train the model first.")
                return

            self.dense_retriever.p_encoder.eval()

            p_embeddings = []
            
            batch_size = self.dense_retriever.args.per_device_eval_batch_size
            
            with torch.no_grad():
                for i in tqdm(range(0, len(self.dense_retriever.contexts), batch_size)):
                    batch = self.dense_retriever.contexts[i:i+batch_size]
                    p_seqs = self.dense_retriever.tokenizer(batch, padding="max_length", truncation=True, return_tensors="pt", max_length=self.dense_retriever.max_len)
                    p_seqs = {key: val.to(self.dense_retriever.device) for key, val in p_seqs.items()}
                
                    embeddings = self.dense_retriever.p_encoder(**p_seqs).cpu()
                    p_embeddings.append(embeddings)
                    
                    del p_seqs
                    torch.cuda.empty_cache()

            self.dense_retriever.p_embeddings = torch.cat(p_embeddings, dim=0)
            with open(emd_path, "wb") as file:
                pickle.dump(self.dense_retriever.p_embeddings, file)
            print("Embedding pickle saved.")
        print("Finished get_dense_embedding_hybrid1 method")
