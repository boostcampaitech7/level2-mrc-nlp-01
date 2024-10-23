import json
import sys
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union
from datetime import datetime

import faiss
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datasets import Dataset, DatasetDict, Features, Sequence, Value, concatenate_datasets, load_from_disk
from tqdm.auto import tqdm
from tqdm import trange
from transformers import AutoModel, AutoTokenizer, TrainingArguments, BertModel, BertPreTrainedModel, AdamW, get_linear_schedule_with_warmup, GPT2LMHeadModel, GPT2TokenizerFast, PreTrainedTokenizerFast, AutoModelForCausalLM
import re
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class BertEncoder(BertPreTrainedModel):
    def __init__(self,
        config
    ):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self,
            input_ids,
            attention_mask=None,
            token_type_ids=None
        ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        return pooled_output

class DenseRetrieval:
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print(f"Padding token: {self.tokenizer.pad_token}")
        print(f"Padding token ID: {self.tokenizer.pad_token_id}")
        
        self.model = AutoModel.from_pretrained(self.model_name)
        self.p_encoder = BertEncoder.from_pretrained(self.model_name).to(self.device)
        self.q_encoder = BertEncoder.from_pretrained(self.model_name).to(self.device)
        self.p_embeddings = None

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
        
    def generate_gpt_negatives(self, contexts, num_negatives=1, batch_size=1):
        model_name = "skt/kogpt2-base-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=1024)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        device = torch.device("cpu")
        model = model.to(device)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        max_length = 512  # 최대 길이를 512로 제한
        negatives = []

        for i in range(0, len(contexts), batch_size):
            batch_contexts = contexts[i:i+batch_size]
            batch_prompts = []
            
            for context in batch_contexts:
                tokenized_context = tokenizer.encode(context, truncation=True, max_length=max_length)
                truncated_context = tokenizer.decode(tokenized_context)
                prompt = f"""다음은 원본 문단의 내용을 변경한 거짓 정보입니다. 원본 문단의 주제와 관련된 내용을 유지하면서 사실을 변경하세요:

원본 문단: {truncated_context}

변경된 문단:"""
                batch_prompts.append(prompt)

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            
            # token_type_ids 제거
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            logger.info(f"Input shape: {inputs['input_ids'].shape}")
            
            with torch.no_grad():
                try:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=300,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        do_sample=True, 
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                    )
                    
                    logger.info(f"Output shape: {outputs.shape}")
                    
                except RuntimeError as e:
                    logger.error(f"Error during generation: {e}")
                    continue

            for j, output in enumerate(outputs):
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                split_text = generated_text.split("변경된 문단:")
                if len(split_text) > 1:
                    modified_context = split_text[-1].strip()
                else:
                    modified_context = generated_text.strip()
                
                if modified_context:
                    negatives.append(modified_context)
                    logger.info(f"Successfully generated negative for context.")
                    logger.info(f"Original context: {batch_contexts[j]}")
                    logger.info(f"Generated negative: {modified_context}")
                else:
                    logger.warning(f"Generated text is empty. Using original context.")
                    negatives.append(batch_contexts[j])

        return negatives

    def prepare_in_batch_negative(self, dataset=None, num_neg=2):
        negative_samples_path = os.path.join(self.data_path, "negative_samples.pkl")
        if os.path.exists(negative_samples_path):
            print("Loading pre-generated negative samples...")
            with open(negative_samples_path, "rb") as f:
                p_with_neg = pickle.load(f)
        else:
            if dataset is None:
                dataset = self.dataset
            if num_neg is None:
                num_neg = self.num_negatives

            p_with_neg = []

            for c in tqdm(dataset["context"], total=len(dataset), desc="Preparing in-batch negatives"):
                p_with_neg.append(c)
                
                # GPT로 네거티브 샘플 생성
                gpt_negatives = self.generate_gpt_negatives([c], num_negatives=num_neg)
                p_with_neg.extend(gpt_negatives)

            print("Tokenizing contexts...")
            p_seqs = self.tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt', max_length=self.max_len)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, self.max_len)
            p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, self.max_len)
            p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, self.max_len)

            print("Creating TensorDataset...")
            train_dataset = TensorDataset(
                p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
            )

            print("Creating DataLoader...")
            self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.per_device_train_batch_size, shuffle=True)
            print("In-batch negative preparation completed.")

            # 네거티브 샘플 저장
            with open(negative_samples_path, "wb") as f:
                pickle.dump(p_with_neg, f)

    def train(self, args=None):
        if args is None:
            args = self.args

        print("Training encoder")

        batch_size = args.per_device_train_batch_size

        self.prepare_in_batch_negative()

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total
        )

        global_step = 0

        self.q_encoder.zero_grad()
        self.p_encoder.zero_grad()
        
        torch.cuda.empty_cache()

        for i in tqdm(range(int(args.num_train_epochs)), desc="Epoch"):
            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    self.q_encoder.train()
                    self.p_encoder.train()

                    targets = torch.zeros(batch_size).long().to(self.device)

                    q_inputs = {
                        "input_ids": batch[0].to(self.device),
                        "attention_mask": batch[1].to(self.device),
                        "token_type_ids": batch[2].to(self.device)
                    }

                    p_inputs = {
                        "input_ids": batch[0].view(batch_size * (self.num_negatives + 1), -1).to(self.device),
                        "attention_mask": batch[1].view(batch_size * (self.num_negatives + 1), -1).to(self.device),
                        "token_type_ids": batch[2].view(batch_size * (self.num_negatives + 1), -1).to(self.device)
                    }

                    q_outputs = self.q_encoder(**q_inputs)
                    p_outputs = self.p_encoder(**p_inputs)

                    q_outputs = q_outputs.view(batch_size, 1, -1)
                    p_outputs = p_outputs.view(batch_size, self.num_negatives + 1, -1)

                    sim_scores = torch.bmm(q_outputs, p_outputs.transpose(1, 2)).squeeze()
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)
                
                    loss = F.nll_loss(sim_scores, targets)
                    
                    tepoch.set_postfix(loss=f"{str(loss.item())}", step=f"{global_step+1}/{t_total}", lr=f"{scheduler.get_last_lr()[0]:.8f}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.q_encoder.zero_grad()
                    self.p_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()
                    del p_inputs, q_inputs

        self.p_encoder.save_pretrained(os.path.join(self.data_path, f"p_encoder"))
        self.q_encoder.save_pretrained(os.path.join(self.data_path, f"q_encoder"))


    def build_faiss(self, num_clusters=64) -> NoReturn:
        """
        FAISS 인덱스를 생성하고 학습시킵니다.
        """
        
        # 인덱스 생성
        quantizer = faiss.IndexFlatL2(self.p_embeddings.shape[1])  
        self.indexer = faiss.IndexIVFScalarQuantizer(quantizer, 
                                                    self.p_embeddings.shape[1],
                                                    num_clusters,
                                                    faiss.METRIC_L2)
        
        # 인덱스 학습
        self.indexer.train(self.p_embeddings.numpy())
        
        # 벡터 추가
        self.indexer.add(self.p_embeddings.numpy())
        
        print('FAISS indexer built')
    
    def get_dense_embedding(self):

        pickle_name = f"dense_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embeddings = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            if os.path.exists(os.path.join(self.data_path, "p_encoder")) and os.path.exists(os.path.join(self.data_path, "q_encoder")):
                self.p_encoder = BertEncoder.from_pretrained(os.path.join(self.data_path, "p_encoder")).to(self.device)
                self.q_encoder = BertEncoder.from_pretrained(os.path.join(self.data_path, "q_encoder")).to(self.device)
            else:
                self.train()

            self.p_encoder.eval()

            p_embeddings = []
            
            batch_size = self.args.per_device_eval_batch_size
            
            with torch.no_grad():
                for i in tqdm(range(0, len(self.contexts), batch_size)):
                    batch = self.contexts[i:i+batch_size]
                    p_seqs = self.tokenizer(batch, padding="max_length", truncation=True, return_tensors="pt", max_length=self.max_len)
                    p_seqs = {key: val.to(self.device) for key, val in p_seqs.items()}
                
                    embeddings = self.p_encoder(**p_seqs).cpu()  # Move to CPU here
                    p_embeddings.append(embeddings)
                    
                    del p_seqs
                    torch.cuda.empty_cache()

            self.p_embeddings = torch.cat(p_embeddings, dim=0)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embeddings, file)
            print("Embedding pickle saved.")


    def retrieve(self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1) -> Union[Tuple[List, List], pd.DataFrame]:
        
        assert self.p_embeddings is not None, "get_dense_embedding() 메소드를 먼저 수행해줘야합니다."

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
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas
        
    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        self.q_encoder.eval()
        self.q_encoder.to(self.device)

        with torch.no_grad():
            q_seqs = self.tokenizer([query], padding="max_length", truncation=True, return_tensors="pt", max_length=self.max_len)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            q_seqs = {key: val.to(self.device) for key, val in q_seqs.items()}
            q_embedding = self.q_encoder(**q_seqs).cpu()

            sim_scores = torch.matmul(q_embedding, self.p_embeddings.transpose(0, 1)).squeeze()
            doc_score, doc_indices = torch.topk(sim_scores, k=k)

        torch.cuda.empty_cache()

        return doc_score.tolist(), doc_indices.tolist()

    def get_relevant_doc_bulk(self, queries: List[str], k: Optional[int] = 1) -> Tuple[List[List[float]], List[List[int]]]:
        self.q_encoder.eval()
        self.q_encoder.to(self.device)

        batch_size = 16  # Adjust this value based on your memory constraints
        q_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(queries), batch_size), desc="Dense retrieval: "):
                batch_queries = queries[i:i+batch_size]
                q_seqs = self.tokenizer(batch_queries, padding="max_length", truncation=True, return_tensors="pt", max_length=self.max_len)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                q_seqs = {key: val.to(self.device) for key, val in q_seqs.items()}
                batch_embeddings = self.q_encoder(**q_seqs).cpu()
                q_embeddings.append(batch_embeddings)

            q_embeddings = torch.cat(q_embeddings, dim=0)

            sim_scores = torch.matmul(q_embeddings, self.p_embeddings.transpose(0, 1))
            doc_score, doc_indices = torch.topk(sim_scores, k=k, dim=1)

        torch.cuda.empty_cache()

        return doc_score.tolist(), doc_indices.tolist()
    
    def retrieve_faiss(self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1) -> Union[Tuple[List, List], pd.DataFrame]:
        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(queries, k=topk)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Dense retrieval: ")):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        self.q_encoder.to(self.device)
        self.q_encoder.eval()

        with torch.no_grad():
            q_seqs = self.tokenizer([query], padding="max_length", truncation=True, return_tensors="pt", max_length=self.max_len)
            q_seqs = {key: val.to(self.device) for key, val in q_seqs.items()}
            q_emb = self.q_encoder(**q_seqs).cpu().numpy()

        
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)
        
        torch.cuda.empty_cache()
        
        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        
        self.q_encoder.to(self.device)
        self.q_encoder.eval()

        batch_size = 16  # 메모리 제약에 따라 이 값을 조정하세요
        D_list = []
        I_list = []

        for i in tqdm(range(0, len(queries), batch_size), desc = "Dense retrieval: "):
            batch_queries = queries[i:i+batch_size]
            
            with torch.no_grad():
                q_seqs = self.tokenizer(batch_queries, padding="max_length", truncation=True, return_tensors="pt", max_length=self.max_len)
                q_seqs = {key: val.to(self.device) for key, val in q_seqs.items()}
                query_vecs = self.q_encoder(**q_seqs).cpu().numpy()
            
            D, I = self.indexer.search(query_vecs, k)
            
            D_list.extend(D.tolist())
            I_list.extend(I.tolist())

            # GPU 메리 리
            torch.cuda.empty_cache()

        return D_list, I_list

    def run(self, datasets, training_args, config):

        self.get_dense_embedding()
        
        if config.dataRetreival.faiss.use(False):
            self.build_faiss(
                num_clusters=config.dataRetreival.faiss.num_clusters(64)
            )
            df = self.retrieve_faiss(
                datasets["validation"],
                topk=config.dataRetreival.top_k(5),
            )
        else:
            df = self.retrieve(
                datasets["validation"],
                topk=config.dataRetreival.top_k(5),
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

    retriever = DenseRetrieval()

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


    if os.path.exists(os.path.join(retriever.data_path, "p_encoder")) and os.path.exists(os.path.join(retriever.data_path, "q_encoder")):
        retriever.p_encoder = BertEncoder.from_pretrained(os.path.join(retriever.data_path, "p_encoder"))
        retriever.q_encoder = BertEncoder.from_pretrained(os.path.join(retriever.data_path, "q_encoder"))
    else:
        retriever.train()
    
    retriever.get_dense_embedding()
    if retriever.config.faiss.use():
        retriever.build_faiss()

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if retriever.config.faiss.use():

        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["correct"] = df.apply(lambda row: row["original_context"] in row["context"], axis=1)

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds, topk=1)
            df["correct"] = df["correct"] = df.apply(lambda row: row["original_context"] in row["context"], axis=1)
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )

        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query, topk=1)

    # 시험적으로 generate_gpt_negatives 메서드 테스트
    test_contexts = [
        "미국 상의원 또는 미국 상원(United States Senate)은 양원제인 미국 의회의 상원이다. 미국 부통령이 상원의장이 된다. 각 주당 2명의 상원의원이 선출되어 100명의 상원의원으로 구성되어 있다. 임기는 6년이며, 2년마다 50개주 중 1/3씩 상원의원을 새로 선출하여 연방에 보낸다.",
        "현대적 경영학의 시작은 1950년대로 볼 수 있다. 2차 세계대전 이후 유럽의 재건과 경제 발전으로 인해 기업의 규모가 커지고 복잡해졌다. 이에 따라 체계적인 경영 이론과 실무가 필요해졌고, 이는 현대적 경영학의 발전으로 이어졌다."
    ]
    
    print("Testing generate_gpt_negatives method:")
    retriever.generate_gpt_negatives(test_contexts, num_negatives=1)




