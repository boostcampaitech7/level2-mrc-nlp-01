import os
import pickle
import numpy as np
from tqdm.auto import tqdm

from NegativeSampler import NegativeSampler

class SparseNegativeSampler(NegativeSampler):
    def __init__(self, corpus):
        self.corpus = np.array(corpus)
    
    def make_sparse_embedding(self, sparse_retriever):
        sparse_retriever.get_sparse_embedding()
        self.retriever = sparse_retriever
        
    def load_neg_indices(self, questions, k=None):
        if k is None:
            k = self.num_negatives + 1
        _, neg_indices = self.sparse_retriever.get_relevant_doc_bulk(questions, topk=k)
        with open(os.path.join(self.data_path, "negatives"), "wb") as file:
            pickle.dump(neg_indices, file)    
        return neg_indices
        
    def get_neg_indicies(self, questions, num_negatives):
        if os.path.exists(os.path.join(self.data_path, "negatives")):
            with open(os.path.join(self.data_path, "negatives"), "rb") as file:
                neg_indices = pickle.load(file)
            if len(neg_indices[0]) < num_negatives + 1:
                print("Saved results do not match the current dataset. Recomputing...")
                return self.load_neg_indices(questions, k=100)
        else:
            return self.load_neg_indices(questions, k=100)
    
    
    def offer(self, row, num_negatives):
        neg_indices = self.get_neg_indicies([row["question"]])[0]
        neg_samples = [self.contexts[neg_indices[j]] for j in range(num_negatives+1) if self.contexts[neg_indices[j]] != row["context"]]
        if len(neg_samples) > num_negatives:
            neg_samples = neg_samples[:num_negatives]
        return {
            "question": row["question"],
            "context": row["context"],
            "negatives": neg_samples
        }
        
    def offer_bulk(self, dataset, num_negatives):
        questions = []
        contexts = []
        p_negs = []
        
        neg_indices = self.get_neg_indicies(dataset["question"])
        for i, neg_idx in enumerate(tqdm(neg_indices, desc="Prepare in-batch negatives")):
            neg_samples = [self.contexts[neg_idx[j]] for j in range(num_negatives+1) if self.contexts[neg_idx[j]] != dataset["context"][i]]
            if len(neg_samples) > num_negatives:
                neg_samples = neg_samples[:num_negatives]
            questions.append(dataset["question"][i])
            contexts.append(dataset["context"][i])
            p_negs.append(neg_samples)
        
        return {
            "question": questions,
            "context": contexts,
            "negatives": p_negs
        }