import json
import os
import pickle
import time
import random
from contextlib import contextmanager


import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import Dataset
from utils import timer
from Retrieval.dense_retrieval import DenseRetriever
from Retrieval.sparse_retrieval import SparseRetriever

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

class HybridRetriever:
    def __init__(self, config, dense_retriever : DenseRetriever, sparse_retriever : SparseRetriever):
        set_seed(config.seed())

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

    def train(self):
        self.dense_retriever.train()
        self.sparse_retriever.train()


    def get_relevant_doc_bulk(self, query, topk):
        # Dense Retriever에서 상위 5개 검색
        dense_scores, dense_indices = self.dense_retriever.get_relevant_doc_bulk(query, k=topk)
        
        # Sparse Retriever에서 상위 5개 검색
        sparse_scores, sparse_indices = self.sparse_retriever.get_relevant_doc_bulk(query, k=topk)

        # 점수 정규화
        dense_scores = [score / max(dense_scores) for score in dense_scores]
        sparse_scores = [score / max(sparse_scores) for score in sparse_scores]
        
        # 결과 합치기 (중복 제거)
        combined_results = {}
        for score, idx in zip(dense_scores, dense_indices):
            combined_results[idx] = max(score, combined_results.get(idx, score))
        for score, idx in zip(sparse_scores, sparse_indices):
            combined_results[idx] = max(score, combined_results.get(idx, score))
        
        # 점수 기준으로 정렬
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
        
        # 상위 5개 반환
        doc_indices = [idx for idx, _ in sorted_results[:topk]]
        doc_scores = [score for _, score in sorted_results[:topk]]
        
        return doc_scores, doc_indices
    
    def retrieve(self, query_or_dataset, topk, concat_context: bool):
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
            ):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": [self.contexts[pid] for pid in doc_indices[idx]]
                }
                if concat_context:
                    tmp["context"] = " ".join(tmp["context"])
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas
        