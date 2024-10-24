import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
from .sparse_retrieval import SparseRetrieval
from .dense_retrieval import DenseRetrieval
from config import Config
import torch.nn.functional as F

class Hybrid1Retriever:
    def __init__(self, config):
        self.config = config
        if config.dataRetrieval.type() != 'hybrid1':
            raise ValueError("This retriever is for hybrid1 type only")
        self.ratio = config.dataRetrieval.hybrid_ratio(0.5)
        
        # Load sparse embedding
        self.sparse_retriever = SparseRetrieval(
            tokenize_fn=AutoTokenizer.from_pretrained(config.model.retriever_tokenizer()).tokenize,
            context_path=config.dataRetrieval.context_path(),
            testing=False,
        )
        self.sparse_retriever.load_sparse_embedding(config.dataRetrieval.sparse_embedding_path())

        # Load dense model
        self.dense_model = AutoModel.from_pretrained(config.dataRetrieval.dense_model_path())
        self.dense_tokenizer = AutoTokenizer.from_pretrained(config.dataRetrieval.dense_model_path())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dense_model.to(self.device)

    def retrieve(self, query_or_dataset, topk=1):
        if isinstance(query_or_dataset, str):
            return self.retrieve_single(query_or_dataset, topk=topk)
        elif isinstance(query_or_dataset, pd.DataFrame):
            return self.retrieve_multi(query_or_dataset, topk=topk)
        else:
            raise ValueError("query_or_dataset must be either str or pd.DataFrame")

    def retrieve_single(self, query: str, topk: int = 1):
        # Sparse retrieval
        sparse_scores, sparse_indices = self.sparse_retriever.get_relevant_doc(query, topk)

        # Dense retrieval
        inputs = self.dense_tokenizer(query, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.dense_model(**inputs)
        query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Compute dense scores (this is a simplified version, you might need to adjust based on your exact dense retrieval implementation)
        dense_scores = np.dot(self.sparse_retriever.contexts_emb, query_embedding.T).squeeze()
        dense_indices = dense_scores.argsort()[::-1][:topk]
        dense_scores = dense_scores[dense_indices]

        # Merge scores
        merged_scores = {}
        for score, idx in zip(sparse_scores, sparse_indices):
            merged_scores[idx] = (1 - self.ratio) * score
        for score, idx in zip(dense_scores, dense_indices):
            merged_scores[idx] = merged_scores.get(idx, 0) + self.ratio * score

        sorted_scores = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
        final_indices = [idx for idx, _ in sorted_scores[:topk]]
        final_scores = [score for _, score in sorted_scores[:topk]]

        return final_scores, final_indices

    def retrieve_multi(self, queries: pd.DataFrame, topk: int = 1):
        total = []
        for _, query in tqdm(queries.iterrows(), total=len(queries), desc="Hybrid retrieval"):
            scores, indices = self.retrieve_single(query['question'], topk)
            contexts = [self.sparse_retriever.contexts[idx] for idx in indices]
            
            tmp = {
                "question": query['question'],
                "id": query['id'],
                "context": " ".join(contexts)
            }
            if 'context' in query and 'answers' in query:
                tmp['original_context'] = query['context']
                tmp['answers'] = query['answers']
            total.append(tmp)

        return pd.DataFrame(total)

def main():
    config = Config()
    
    if config.dataRetrieval.type() != 'hybrid1':
        raise ValueError("This script is for hybrid1 type only")
    
    # Load datasets
    datasets = load_from_disk(config.dataQA.path.train('./data/train_dataset'))
    
    # Initialize hybrid retriever
    hybrid_retriever = HybridRetriever(config)
    
    # Perform retrieval
    validation_dataset = datasets["validation"]
    k = config.dataRetrieval.top_k(5)
    df = hybrid_retriever.retrieve(validation_dataset, topk=k)
    
    # Evaluate results
    rankings = []
    for _, row in df.iterrows():
        if row['original_context'] in row['context']:
            rank = row['context'].split(' ').index(row['original_context']) + 1
            rankings.append(min(rank, k))
        else:
            rankings.append(k + 1)
    
    def recall_at_k(k):
        return sum([1 for rank in rankings if rank <= k]) / len(rankings)
    
    recalls = {f"recall@{i}": recall_at_k(i) for i in range(1, k+1)}
    
    # Save results
    output_path = config.output.train('./outputs/train_dataset')
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'hybrid_recalls.json'), 'w') as f:
        json.dump(recalls, f, indent=4)
    
    print(f"Hybrid Recall@{k}: {recalls[f'recall@{k}']}")

if __name__ == "__main__":
    main()
