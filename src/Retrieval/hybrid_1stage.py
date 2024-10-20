import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from tqdm.auto import tqdm

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
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        if not 0 <= ratio <= 1:
            raise ValueError("ratio must be between 0 and 1")

        """
        Arguments:
            sparse_retriever:
                SparseRetrieval입니다.
            dense_retriever:
                DenseRetrieval입니다.
            ratio:
                sparse와 dense 결과의 가중치입니다. (0 <= ratio <= 1)
            context_path:
                Passage들이 묶여있는 파일명입니다.

        Summary:
            SparseRetrieval과 DenseRetrieval을 결합하여 사용하는 클래스입니다.
        """

        self.sparse_retriever = sparse_retriever        
        self.dense_retriever = dense_retriever
        self.sparse_retriever.get_sparse_embedding()
        self.dense_retriever.get_dense_embedding()
        self.ratio = ratio

        data_path = os.path.dirname(context_path)
        context_path = os.path.basename(context_path)
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))


    def retrieve(self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1, use_faiss: bool = False) -> Union[Tuple[List, List], pd.DataFrame]:
        print(f"Hybrid retrieve called with ratio: {self.ratio}")
        if isinstance(query_or_dataset, str):#쿼리가 1개일 떄
            return self.retrieve_single(query_or_dataset, topk=topk, use_faiss=use_faiss)
        elif isinstance(query_or_dataset, Dataset):#쿼리가 다수일 때
            return self.retrieve_multi(query_or_dataset, topk=topk, use_faiss=use_faiss)

    def retrieve_single(self, query: str, topk: Optional[int] = 1, use_faiss: bool = False) -> Tuple[List, List]:
        if use_faiss:
            sparse_scores, sparse_indices = self.sparse_retriever.retrieve_faiss(query, topk=topk)
            dense_scores, dense_indices = self.dense_retriever.retrieve_faiss(query, topk=topk)
        else:
            sparse_scores, sparse_indices = self.sparse_retriever.retrieve(query, topk=topk)
            dense_scores, dense_indices = self.dense_retriever.retrieve(query, topk=topk)
        
        merged_scores, merged_indices = self._merge_results(sparse_scores, sparse_indices, dense_scores, dense_indices, topk)
        
        return merged_scores, [self.contexts[i] for i in merged_indices]


    def retrieve_multi(self, dataset: Dataset, topk: Optional[int] = 1, use_faiss: bool = False) -> pd.DataFrame:
        total = []
        with timer("query hybrid search"):
            for example in tqdm(dataset, desc="Hybrid retrieval: "):
                scores, indices = self.retrieve_single(example["question"], topk=topk, use_faiss=use_faiss)
                
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join([self.contexts[i] for i in indices])
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

        return pd.DataFrame(total)

    def _merge_results(self, sparse_scores, sparse_indices, dense_scores, dense_indices, topk):
        print(f"병합 결과 비율: {self.ratio}")
        print(f"희소 점수 (처음 5개): {sparse_scores[:5]}")
        print(f"밀집 점수 (처음 5개): {dense_scores[:5]}")

        merged_dict = {}
        for score, idx in zip(sparse_scores, sparse_indices):
            merged_dict[idx] = self.ratio * score
        for score, idx in zip(dense_scores, dense_indices):
            merged_dict[idx] = merged_dict.get(idx, 0) + (1 - self.ratio) * score

        merged_items = sorted(merged_dict.items(), key=lambda x: x[1], reverse=True)[:topk]
        merged_indices, merged_scores = zip(*merged_items)
        
        print(f"병합된 점수 (처음 5개): {list(merged_scores)[:5]}")
        print(f"병합된 인덱스 (처음 5개): {list(merged_indices)[:5]}")
        return list(merged_scores), list(merged_indices)

    def run(self, datasets, training_args, config):
        self.sparse_retriever.get_sparse_embedding()
        self.dense_retriever.get_dense_embedding()
        
        use_faiss = config.dataRetrieval.faiss.use(False)
        if use_faiss:
            self.sparse_retriever.build_faiss(num_clusters=config.dataRetrieval.faiss.num_clusters(64))
            self.dense_retriever.build_faiss(num_clusters=config.dataRetrieval.faiss.num_clusters(64))
        
        df = self.retrieve(
            datasets["validation"],
            topk=config.dataRetrieval.top_k(5),
            use_faiss=use_faiss
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
