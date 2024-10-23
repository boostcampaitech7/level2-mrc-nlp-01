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
    def __init__(self, config, dense_retriever, sparse_retriever):
        set_seed(config.seed())
        self.context_path = config.dataRetrieval.context_path()
        with open(self.context_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로

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

        # 점수 정규화 (이중 리스트 처리)
        dense_scores = [[score / max(scores) for score in scores] for scores in dense_scores]
        sparse_scores = [[score / max(scores) for score in scores] for scores in sparse_scores]
        
        # 점수 결합 (행 단위로 처리)
        combined_results = []
        for dense_score, dense_idx, sparse_score, sparse_idx in zip(dense_scores, dense_indices, sparse_scores, sparse_indices):
            combined_result = {}
            for j in range(len(dense_idx)):
                idx = dense_idx[j]
                if idx in combined_result:
                    combined_result[idx] += dense_score[j]
                else:
                    combined_result[idx] = dense_score[j]
            
            for j in range(len(sparse_idx)):
                idx = sparse_idx[j]
                if idx in combined_result:
                    combined_result[idx] += sparse_score[j]
                else:
                    combined_result[idx] = sparse_score[j]
            
            combined_results.append(combined_result)
        
        # 점수 기준으로 정렬 (행 단위로 처리)
        sorted_results = [sorted(result.items(), key=lambda x: x[1], reverse=True) for result in combined_results]
        
        # 상위 k개 결과 반환 (행 단위로 처리)
        doc_scores = [[score for idx, score in result[:topk]] for result in sorted_results]
        doc_indices = [[idx for idx, score in result[:topk]] for result in sorted_results]

        return doc_scores, doc_indices
    
    def retrieve(self, query_or_dataset, topk, concat_context: bool):
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, topk=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], topk=topk
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
        