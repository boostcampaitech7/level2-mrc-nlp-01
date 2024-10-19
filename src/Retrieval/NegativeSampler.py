import numpy as np

class NegativeSampler:
    def __init__(self, num_negatives, corpus):
        self.num_negatives = num_negatives
        self.corpus = np.array(corpus)
        
    def offer(self, row):
        context = row["context"]
        question = row["question"]
        
        while True:
            neg_idxs = np.random.randint(len(self.corpus), size=self.num_negatives)
            if not context in self.corpus[neg_idxs]:
                p_neg = self.corpus[neg_idxs]

                return {
                    "question": question,
                    "context": context,
                    "negatives": p_neg
                }

    def offer_bulk(self, dataset):
        questions = []
        contexts = []
        p_negs = []
        
        for row in dataset:
            sample = self.offer(row)
            questions.append(sample["question"])
            contexts.append(sample["context"])
            p_negs.append(sample["negatives"])
        
        return {
            "question": questions,
            "context": contexts,
            "negatives": p_negs
        }