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
from dense_retrieval import DenseRetrieval 

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
        if isinstance(query_or_dataset, str):#쿼리가 1개일 떄
            return self.retrieve_single(query_or_dataset, topk=topk, use_faiss=use_faiss)
        elif isinstance(query_or_dataset, Dataset):#쿼리가 다수일 때
            return self.retrieve_multi(query_or_dataset, topk=topk, use_faiss=use_faiss)

    def retrieve_single(self, query: str, topk: Optional[int] = 1, use_faiss: bool = False) -> Tuple[List, List]:
        if use_faiss:#faiss 사용 여부
            sparse_scores, sparse_indices = self.sparse_retriever.retrieve_faiss(query, topk=topk)
            dense_scores, dense_indices = self.dense_retriever.retrieve_faiss(query, topk=topk)
        else:#faiss 사용 안할 때
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
        merged_dict = {}
        for score, idx in zip(sparse_scores, sparse_indices):
            merged_dict[idx] = self.ratio * score
        for score, idx in zip(dense_scores, dense_indices):
            merged_dict[idx] = merged_dict.get(idx, 0) + (1 - self.ratio) * score

        merged_scores = sorted(merged_dict.items(), key=lambda x: x[1], reverse=True)[:topk]
        return [score for _, score in merged_scores], [idx for idx, _ in merged_scores]

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

if __name__ == "__main__":
    import argparse
    from datasets import load_from_disk, concatenate_datasets
    from transformers import AutoTokenizer
    from dense_retrieval import DenseRetrieval, BertEncoder
    from sparse_retrieval import SparseRetrieval

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", metavar="./data/train_dataset", type=str, help="")
    parser.add_argument("--model_name_or_path", metavar="bert-base-multilingual-cased", type=str, help="")
    parser.add_argument("--data_path", metavar="./data", type=str, help="")
    parser.add_argument("--context_path", metavar="wikipedia_documents", type=str, help="")
    parser.add_argument("--use_faiss", action="store_true", help="Use FAISS for retrieval")
    parser.add_argument("--ratio", type=float, default=0.5, help="Ratio for sparse and dense results")

    args = parser.parse_args()

    # 데이터셋 로드
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets([
        org_dataset["train"].flatten_indices(),
        org_dataset["validation"].flatten_indices(),
    ])
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    
    sparse_retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path
    )

    dense_retriever = DenseRetrieval(
        model_name=args.model_name_or_path,
        context_path=args.context_path,
    )

    retriever = SparseDenseRetrieval(
        sparse_retriever=sparse_retriever,
        dense_retriever=dense_retriever,
        ratio=args.ratio,
        context_path=args.context_path,
    )

    sparse_retriever.get_sparse_embedding()
    if os.path.exists(os.path.join(dense_retriever.data_path, "p_encoder")) and os.path.exists(os.path.join(dense_retriever.data_path, "q_encoder")):
        dense_retriever.p_encoder = BertEncoder.from_pretrained(os.path.join(dense_retriever.data_path, "p_encoder"))
        dense_retriever.q_encoder = BertEncoder.from_pretrained(os.path.join(dense_retriever.data_path, "q_encoder"))
    else:
        dense_retriever.train()
    
    dense_retriever.get_dense_embedding()
    if args.use_faiss:
        sparse_retriever.build_faiss()
        dense_retriever.build_faiss()

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    print("\n", "=" * 40, "Single Query Test", "=" * 40)
    with timer("single query"):
        scores, indices = retriever.retrieve(query, use_faiss=args.use_faiss)
    print(f"Query: {query}")
    for i, (score, context) in enumerate(zip(scores[:5], indices[:5])):
        print(f"Top-{i+1} passage with score {score:.4f}")
        print(f"Context: {context[:50]}...\n")

    print("\n", "=" * 40, "Bulk Query Test", "=" * 40)
    with timer("bulk query"):
        df = retriever.retrieve(full_ds, use_faiss=args.use_faiss)
        df["correct"] = df["original_context"] == df["context"]
        accuracy = df["correct"].sum() / len(df)
        print(f"Correct retrieval result: {accuracy:.4f}")

    print("\nRetrieval test completed.")
