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

from sparse_retrieval import SparseRetrieval
from dense_retrieval import DenseRetrieval  # Dense 검색 클래스가 이미 구현되어 있다고 가정

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class SparseDenseRetrieval:
    def __init__(
        self,
        sparse_retriever: SparseRetrieval,
        dense_retriever: DenseRetrieval,
        ratio: float = 0.5,
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

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


    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `retrieve_single`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `retrieve_multi`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame
        """

        if isinstance(query_or_dataset, str):
            return self.retrieve_single(query_or_dataset, topk=topk)

        elif isinstance(query_or_dataset, Dataset):
            return self.retrieve_multi(query_or_dataset, topk=topk)


    def retrieve_single(self, query: str, topk: Optional[int] = 1) -> Tuple[List, List]:
        sparse_scores, sparse_indices = self.sparse_retriever.retrieve(query, topk=topk)
        dense_scores, dense_indices = self.dense_retriever.retrieve(query, topk=topk)
        
        merged_scores, merged_indices = self._merge_results(sparse_scores, sparse_indices, dense_scores, dense_indices, topk)
        
        return merged_scores, [self.contexts[i] for i in merged_indices]

    def retrieve_multi(self, dataset: Dataset, topk: Optional[int] = 1) -> pd.DataFrame:
        # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
        total = []
        with timer("query hybrid search"):
            for example in tqdm(dataset, desc="Hybrid retrieval: "):
                sparse_scores, sparse_indices = self.sparse_retriever.retrieve(example["question"], topk=topk)
                dense_scores, dense_indices = self.dense_retriever.retrieve(example["question"], topk=topk)
                
                merged_scores, merged_indices = self._merge_results(sparse_scores, sparse_indices, dense_scores, dense_indices, topk)
                
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join([self.contexts[i] for i in merged_indices])
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

        return pd.DataFrame(total)
    

    def _merge_results(self, sparse_scores, sparse_indices, dense_scores, dense_indices, topk):
        all_indices = list(set(sparse_indices + dense_indices))
        merged_scores = []

        for idx in all_indices:
            sparse_score = sparse_scores[sparse_indices.index(idx)] if idx in sparse_indices else 0
            dense_score = dense_scores[dense_indices.index(idx)] if idx in dense_indices else 0
            merged_score = self.ratio * sparse_score + (1 - self.ratio) * dense_score
            merged_scores.append((merged_score, idx))

        merged_scores.sort(reverse=True)
        return [score for score, _ in merged_scores[:topk]], [idx for _, idx in merged_scores[:topk]]





# 메인 실행 부분은 필요에 따라 추가할 수 있습니다.