import json
import os
import pickle
import time
import random
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi
from concurrent.futures import ThreadPoolExecutor, as_completed

seed = 2024
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정



@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        context_path: Optional[str] = "./data/wikipedia_documents.json",
        testing: bool = False
    ) -> NoReturn:
        self.testing = testing
        data_path = os.path.dirname(context_path)
        context_path = os.path.basename(context_path)
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        if self.testing:
            total_documents = len(wiki)
            num_documents = int(0.01 * total_documents)
            wiki = {k: wiki[k] for k in list(wiki.keys())[:num_documents]}

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.p_embedding = None
        self.indexer = None
        self.bm25 = None
        self.tokenize_fn = tokenize_fn


    def parallel_tokenize(self, contexts: List[str]) -> List[List[str]]:
        """
        여러 문서를 병렬로 토큰화합니다.
        """
        tokenized_contexts = []
        
        # ThreadPoolExecutor를 사용하여 병렬로 토큰화
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.tokenize_fn, doc) for doc in contexts]
            
            # 토큰화 결과를 순서대로 리스트에 저장
            for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel tokenizing"):
                tokenized_contexts.append(future.result())
        
        return tokenized_contexts


    def train(self, emb_path=None):
        if emb_path is None:
            pickle_name = f"bm25_sparse_embedding.bin"
            emb_path = os.path.join(self.data_path, pickle_name)
        
        # 병렬 토큰화 사용
        tokenized_contexts = self.parallel_tokenize(self.contexts)
        
        # BM25 학습
        self.bm25 = BM25Okapi(tokenized_contexts)
        
        # BM25 모델 저장
        with open(emb_path, "wb") as file:
            pickle.dump(self.bm25, file)
        
        print("-----------BM25 pickle saved.-----------")
        
        

    def get_sparse_embedding(self) -> NoReturn:
        """
        BM25 Embedding을 생성하고, pickle로 저장된 것이 있으면 불러옵니다.
        """
        pickle_name = f"bm25_sparse_embedding.bin"
        emb_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emb_path):
            with open(emb_path, "rb") as file:
                self.bm25 = pickle.load(file)
            print("-----------BM25 pickle loaded.-----------")
        else:
            print("Build BM25 embedding")
            self.train(emb_path)
            
    def build_faiss(self, num_clusters=64) -> NoReturn:

        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1, concat_context: bool = True
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        # assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."
        # BM25는 p_embedding 필요 없음. BM25는 쿼리가 주어질 때마다 점수를 계산하기 때문...

        if isinstance(query_or_dataset, str):
            # 단일 쿼리 처리 (변경 없음)
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # 저장된 결과 파일 이름 (예: 'retrieval_results.pkl')
            results_filename = "retrieval_results.pkl"

            # 저장된 결과가 있는지 확인
            if os.path.exists(results_filename):
                with open(results_filename, 'rb') as f:
                    doc_scores, doc_indices = pickle.load(f)
                print("Loaded pre-computed retrieval results.")
                
                # 저장된 결과의 길이 확인
                if len(doc_indices) != len(query_or_dataset):
                    print("Saved results do not match the current dataset. Recomputing...")
                    compute_new_results = True
                else:
                    compute_new_results = False
            else:
                compute_new_results = True

            if compute_new_results:
                # 결과를 새로 계산
                with timer("query exhaustive search"):
                    doc_scores, doc_indices = self.get_relevant_doc_bulk(
                        query_or_dataset["question"], k=topk
                    )
                
                # 결과 저장
                with open(results_filename, 'wb') as f:
                    pickle.dump((doc_scores, doc_indices), f)
                print("Computed and saved retrieval results.")

            # 디버깅을 위한 출력 추가
            print("Shape of doc_indices:", np.array(doc_indices).shape)
            print("Length of query_or_dataset:", len(query_or_dataset))

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                if idx >= len(doc_indices):
                    print(f"Error: idx {idx} is out of range for doc_indices")
                    break
                # print('debug:', idx, example)
                # print('doc_indices:', doc_indices)
                # print('doc_indices shape:', np.array(doc_indices).shape)
                # print('doc_indices for this example:', doc_indices[idx])
                # print('self.contexts length:', len(self.contexts))
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": [self.contexts[pid] for pid in doc_indices[idx]]
                }
                if concat_context:
                    tmp["context"] = " ".join(tmp["context"])
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas
    
    def get_relevant_doc(
        self, query: str, k: Optional[int] = 1
        ) -> Tuple[List, List]:
        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        """
        # self.bm25가 None인 경우 BM25 모델을 로드하거나 학습합니다.
        if self.bm25 is None:
            self.get_sparse_embedding()

        tokenized_query = self.tokenize_fn(query)
        scores = self.bm25.get_scores(tokenized_query)
        sorted_result = np.argsort(scores)[::-1]

        doc_scores, doc_indices = scores[sorted_result][:k].tolist(), sorted_result[:k].tolist()
        return doc_scores, doc_indices

    def process_single_query(self, query):
        tokenized_query = self.tokenize_fn(query)
        scores = self.bm25.get_scores(tokenized_query)
        sorted_result = np.argsort(scores)[::-1]
        return scores[sorted_result], sorted_result
    
    
    def get_relevant_doc_bulk(self, queries: List[str], k: Optional[int] = 1) -> Tuple[List[List[float]], List[List[int]]]:
        if self.bm25 is None:
            self.get_sparse_embedding()

        doc_scores = []
        doc_indices = []
        print('get_relevant_doc_bulk 실행중 queries의 개수는 ', len(queries))
        print('get_relevant_doc_bulk 실행중 queries의 type은 ', type(queries))

        with ThreadPoolExecutor() as executor:
            print("self.process_single_query하는중")
            future_to_query = {executor.submit(self.process_single_query, query): query for query in queries}
            print("self.process_single_query끝남")
            for future in tqdm(as_completed(future_to_query), total=len(future_to_query), desc="Processing queries"):
                scores, indices = future.result()
                doc_scores.append(scores[:k].tolist())
                doc_indices.append(indices[:k].tolist())
            
        return doc_scores, doc_indices
        
    
    
    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vecs = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()

    def run(self, datasets, training_args, config):
        self.get_sparse_embedding()
        
        if config.dataRetrieval.faiss.use(False):
            self.build_faiss(
                num_clusters=config.dataRetrieval.faiss.num_clusters(64)
            )
            df = self.retrieve_faiss(
                datasets["validation"],
                topk=config.dataRetrieval.top_k(5),
            )
        else:
            df = self.retrieve(
                datasets["validation"],
                topk=config.dataRetrieval.top_k(5),
            )
        
        if training_args.do_predict:
            f = Features(
                {
                    "context": Value(dtype="string", id=None),
                    "id": Value(dtype="string", id=None),
                    "question": Value(dtype="string", id=None),
                }
            )
        
            datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
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
        
            datasets = DatasetDict({"validation": Dataset.from_pandas(df)})
        return datasets


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", metavar="./data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        metavar="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", metavar="./data", type=str, help="")
    parser.add_argument(
        "--context_path", metavar="wikipedia_documents", type=str, help=""
    )
    parser.add_argument("--use_faiss", metavar=False, type=bool, help="")

    args = parser.parse_args()

    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False,)
    
    retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        context_path=args.context_path,
    )

    retriever.get_sparse_embedding()
    if args.use_faiss:
        retriever.build_faiss()

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:

        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds)
            df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )

        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query)
