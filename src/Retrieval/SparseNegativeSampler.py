import os
import pickle
import numpy as np
from datasets import Dataset, concatenate_datasets
from tqdm.auto import tqdm

from NegativeSampler import NegativeSampler

class SparseNegativeSampler(NegativeSampler):
    def __init__(self, corpus, data_path="data"):
        self.data_path = data_path
        self.corpus = np.array(corpus)
    
    def make_sparse_embedding(self, sparse_retriever):
        sparse_retriever.get_sparse_embedding()
        self.retriever = sparse_retriever
        
    def load_samples(self, dataset, k=None, save=True):
        _, neg_indices = self.retriever.get_relevant_doc_bulk(dataset["question"], k=k)
        sample_dataset = Dataset.from_dict({"id": dataset["id"], "sample_indices": neg_indices})
        if save:
            print("hi")
            with open(os.path.join(self.data_path, "samples"), "wb") as file:
                pickle.dump(sample_dataset, file)   
        return sample_dataset

    def get_samples(self, dataset, num_negatives):
        if os.path.exists(os.path.join(self.data_path, "samples")):
            with open(os.path.join(self.data_path, "samples"), "rb") as file:
                sample_dataset = pickle.load(file)
            # load된 dataset이 sampling이 덜 되었다면
            if len(sample_dataset["sample_indices"][0]) < num_negatives + 1:
                print("Saved results do not match the current dataset. Recomputing...")
                sample_dataset = self.load_samples(dataset, k=num_negatives+1)
        else:
            sample_dataset = self.load_samples(dataset, k=1+max(num_negatives, 100))
            
        sampled_ids = set(sample_dataset["id"])
        queried_ids = set(dataset["id"])
        queried_but_not_sampled_ids = queried_ids - sampled_ids
        
        # load된 dataset에 존재하지 않은 id가 ids에 주어졌다면
        if len(queried_but_not_sampled_ids) > 0:
            not_sampled_dataset = dataset.filter(lambda row: row["id"] in queried_but_not_sampled_ids)
            num_samples = len(sample_dataset["sample_indices"][0])
            additional_sampled_dataset = self.load_samples(not_sampled_dataset, k=num_samples, save=False)
            
            concated_dataset = concatenate_datasets([sample_dataset, additional_sampled_dataset])
            with open(os.path.join(self.data_path, "samples"), "wb") as file:
                pickle.dump(concated_dataset, file)   
            
            sample_dataset = concated_dataset
        
        # print(sample_dataset)
        filtered_dataset = sample_dataset.filter(lambda row: row['id'] in queried_ids)
        
        return list(map(lambda indices: indices[:num_negatives+1],filtered_dataset["sample_indices"]))
    
    def offer(self, row, num_negatives, exclude_positive=True):
        neg_indices = self.get_samples(Dataset.from_dict(row), num_negatives)[0]
        if exclude_positive:
            neg_samples = [self.corpus[neg_indices[j]] for j in range(num_negatives+1) if self.corpus[neg_indices[j]] != row["context"]]
            if len(neg_samples) > num_negatives:
                neg_samples = neg_samples[:num_negatives]
        else:
            neg_indices = neg_indices[:num_negatives]
            neg_samples = self.corpus[neg_indices]
        return {
            "question": row["question"],
            "context": row["context"],
            "negatives": neg_samples
        }
        
    def offer_bulk(self, dataset, num_negatives, exclude_positive=True):
        questions = []
        contexts = []
        p_negs = []
        
        neg_indices = self.get_samples(dataset, num_negatives)
        for i, neg_idx in enumerate(tqdm(neg_indices, desc="Prepare in-batch negatives")):
            # neg_samples = [self.corpus[neg_idx[j]] for j in range(num_negatives+1) if self.corpus[neg_idx[j]] != dataset["context"][i]]
            neg_samples = self.corpus[neg_idx]
            if exclude_positive:
                neg_samples = neg_samples[neg_samples != dataset["context"][i]]
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