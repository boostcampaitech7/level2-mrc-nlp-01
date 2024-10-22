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

from .sparse_retrieval import SparseRetrieval
from .dense_retrieval import DenseRetrieval

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class hybrid_1stage:
    def __init__(
        self,
        sparse_retriever: SparseRetrieval,
        dense_retriever: DenseRetrieval,
        ratio: float = 0.5,
        context_path: Optional[str] = "./data/wikipedia_documents.json",
    ) -> NoReturn:
        if not 0 <= ratio <= 1:
            raise ValueError("ratio must be between 0 and 1")
        
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.ratio = ratio
        self.context_path = context_path
        self.contexts = None
        
        self.load_contexts()
        
        # BM25 모델 초기화 확인
        if self.sparse_retriever.bm25 is None:
            print("Initializing BM25 model...")
            self.sparse_retriever.get_sparse_embedding()
        
        # Dense 임베딩 로드 확인
        if self.dense_retriever.p_embeddings is None:
            print("Loading dense embeddings...")
            self.dense_retriever.get_dense_embedding()
        
        print(f"Initialized hybrid_1stage with ratio: {self.ratio}")  # 로그 추가
    
    def load_contexts(self):
        if self.contexts is None:
            with open(self.context_path, "r", encoding="utf-8") as f:
                wiki = json.load(f)
            self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
            print(f"Loaded {len(self.contexts)} unique contexts")

    def retrieve(self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1) -> Union[Tuple[List, List], pd.DataFrame]:
        # BM25 모델 재확인
        if self.sparse_retriever.bm25 is None:
            print("BM25 model is not initialized. Initializing now...")
            self.sparse_retriever.get_sparse_embedding()
        
        # Dense 임베딩 재확인
        if self.dense_retriever.p_embeddings is None:
            print("Dense embeddings are not loaded. Loading now...")
            self.dense_retriever.get_dense_embedding()
        
        print(f"Hybrid retrieve called with ratio: {self.ratio}")
        if isinstance(query_or_dataset, str):
            return self.retrieve_single(query_or_dataset, topk=topk)
        elif isinstance(query_or_dataset, Dataset):
            return self.retrieve_multi(query_or_dataset, topk=topk)
        else:
            sparse_scores, sparse_indices = self.sparse_retriever.retrieve(query, topk=topk)
            dense_scores, dense_indices = self.dense_retriever.retrieve(query, topk=topk)
        
        merged_scores, merged_indices = self._merge_results(sparse_scores, sparse_indices, dense_scores, dense_indices, topk)
        
        return merged_scores, [self.contexts[i] for i in merged_indices]


    def retrieve_multi(self, dataset: Dataset, topk: Optional[int] = 1, use_faiss: bool = False) -> pd.DataFrame:
        total = []
        with tqdm(total=len(queries)) as pbar:
            for i, query in enumerate(queries):
                question = query['question']
                scores, indices = self.retrieve_single(question, topk)
                
                context = " ".join([self.contexts[idx] for idx in indices])
                
                tmp = {
                    "question": question,
                    "id": query["id"],
                    "context": context
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
            merged_dict[self._get_index(idx)] = self.ratio * score
        for score, idx in zip(dense_scores, dense_indices):
            merged_dict[self._get_index(idx)] = merged_dict.get(self._get_index(idx), 0) + (1 - self.ratio) * score

        merged_items = sorted(merged_dict.items(), key=lambda x: x[1], reverse=True)[:topk]
        
        if not merged_items:
            print("경고: 병합된 결과가 없습니다.")
            return [], []
        
        merged_indices, merged_scores = zip(*merged_items)
        
        print(f"병합된 점수 (처음 5개): {list(merged_scores)[:5]}")
        print(f"병합된 인덱스 (처음 5개): {list(merged_indices)[:5]}")
        return list(merged_scores), list(merged_indices)

    def run(self, datasets: DatasetDict, training_args: TrainingArguments, config: Dict) -> DatasetDict:
        print("Starting hybrid_1stage run method")
        if training_args.do_eval or training_args.do_predict:
            print(f"Initial ratio: {self.ratio}")
            self.dense_retriever.search_results_cache = None
            self.sparse_retriever.search_results_cache = None

            new_ratio = config.dataRetrieval.hybrid_ratio(0.5)
            print(f"Config hybrid_ratio: {new_ratio}")
            self.set_ratio(new_ratio)
            print(f"Updated hybrid ratio to: {self.ratio}")
            
            top_k = config.dataRetrieval.top_k(1)
            print(f"Config top_k: {top_k}")
            print(f"Using hybrid ratio: {self.ratio}")
            print(f"Using top_k: {top_k}")

            for split in datasets.keys():
                print(f"\nProcessing {split} set")
                try:
                    df = self.retrieve(datasets[split], topk=top_k)
                    print(f"Retrieved {len(df)} results for {split} set")
                    print(f"Sample results:\n{df.head()}")
                    
                    f = Features({
                        "context": Value(dtype="string", id=None),
                        "id": Value(dtype="string", id=None),
                        "question": Value(dtype="string", id=None),
                    })
                    
                    if "answers" in df.columns:
                        f["answers"] = Sequence(feature={
                            "text": Value(dtype="string", id=None),
                            "answer_start": Value(dtype="int32", id=None),
                        }, length=-1, id=None)

                    datasets[split] = Dataset.from_pandas(df, features=f)
                    print(f"Created dataset for {split} set with {len(datasets[split])} examples")
                    print(f"Sample from new dataset:\n{datasets[split][:5]}")
                except Exception as e:
                    print(f"Error processing {split} set: {str(e)}")
                    import traceback
                    traceback.print_exc()

        print("Finished hybrid_1stage run method")
        return datasets

    def set_ratio(self, new_ratio: float):
        if not 0 <= new_ratio <= 1:
            raise ValueError("ratio must be between 0 and 1")
        self.ratio = new_ratio
        print(f"Ratio updated to: {self.ratio}")
