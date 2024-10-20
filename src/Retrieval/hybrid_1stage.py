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


    def retrieve(self, query_or_dataset, topk=1, use_faiss=False):
        if isinstance(query_or_dataset, str):
            # 단일 쿼리 처리
            sparse_scores, sparse_indices = self.sparse_retriever.retrieve(query_or_dataset, topk=topk)
            dense_scores, dense_indices = self.dense_retriever.retrieve(query_or_dataset, topk=topk)
            scores, indices = self._merge_results(sparse_scores, sparse_indices, dense_scores, dense_indices, topk)
            
            print(f"self.contexts의 길이: {len(self.contexts)}")
            print(f"반환된 indices: {indices}")
            
            # indices가 비어 있는 경우 처리
            if not indices:
                print("경고: 검색 결과가 없습니다.")
                return [], []
            
            # indices가 문자열인 경우 (이미 context인 경우) 그대로 반환
            if isinstance(indices[0], str):
                return scores, indices
            
            # indices가 정수인 경우 context로 변환, 유효한 인덱스만 사용
            valid_contexts = []
            valid_scores = []
            for score, idx in zip(scores, indices):
                try:
                    idx_int = int(idx)
                    if 0 <= idx_int < len(self.contexts):
                        valid_contexts.append(self.contexts[idx_int])
                        valid_scores.append(score)
                    else:
                        print(f"경고: 인덱스 {idx_int}가 contexts의 범위를 벗어났습니다.")
                except ValueError:
                    print(f"경고: 인덱스 '{idx}'를 정수로 변환할 수 없습니다.")
            
            return valid_scores, valid_contexts
        else:
            # 데이터셋 처리
            return self.retrieve_multi(query_or_dataset, topk=topk)

    def retrieve_multi(self, dataset, topk=1):
        total = []
        with timer("query hybrid search"):
            for example in tqdm(dataset, desc="Hybrid retrieval: "):
                scores, contexts = self.retrieve(example["question"], topk=topk)
                
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join(contexts)
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
            merged_dict[self._get_index(idx)] = self.ratio * score
        for score, idx in zip(dense_scores, dense_indices):
            merged_dict[self._get_index(idx)] = merged_dict.get(self._get_index(idx), 0) + (1 - self.ratio) * score

        merged_items = sorted(merged_dict.items(), key=lambda x: x[1], reverse=True)[:topk]
        
        if not merged_items:
            print("경고: 병합된 결과가 없습니다.")
            return [], []
        
        merged_indices, merged_scores = zip(*merged_items)
        
        # 유효한 인덱스만 필터링
        valid_indices = [idx for idx in merged_indices if 0 <= self._get_index(idx) < len(self.contexts)]
        valid_scores = [score for idx, score in zip(merged_indices, merged_scores) if 0 <= self._get_index(idx) < len(self.contexts)]
        
        print(f"병합된 점수 (처음 5개): {valid_scores[:5]}")
        print(f"병합된 인덱스 (처음 5개): {valid_indices[:5]}")
        
        return valid_scores, valid_indices

    def _get_index(self, idx):
        if isinstance(idx, int):
            return idx
        elif isinstance(idx, str):
            # 문자열이 숫자로만 구성되어 있는지 확인
            if idx.isdigit():
                return int(idx)
            else:
                # 숫자가 아닌 경우, 해시 값을 사용하여 고유한 정수 생성
                return hash(idx)
        else:
            # 다른 타입의 경우 해시 값 사용
            return hash(str(idx))

    def run(self, datasets, training_args, config):
        self.sparse_retriever.get_sparse_embedding()
        self.dense_retriever.get_dense_embedding()
        
        use_faiss = config.dataRetrieval.faiss.use(False)
        if use_faiss:
            self.sparse_retriever.build_faiss(num_clusters=config.dataRetrieval.faiss.num_clusters(64))
            self.dense_retriever.build_faiss(num_clusters=config.dataRetrieval.faiss.num_clusters(64))
        
        df = self.retrieve(
            datasets["validation"],
            topk=config.dataRetrieval.top_k(5)
        )
        
        # 누락된 열 처리
        if 'original_context' not in df.columns:
            df['original_context'] = ''
        if 'answers' not in df.columns:
            df['answers'] = [{'text': [], 'answer_start': []}] * len(df)
        
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
                    "id": Value(dtype="string", id=None),
                    "question": Value(dtype="string", id=None),
                    "original_context": Value(dtype="string", id=None),
                }
            )
        
        # 데이터프레임에서 필요한 열만 선택
        columns_to_use = list(f.keys())
        df = df[columns_to_use]
        
        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        return datasets
