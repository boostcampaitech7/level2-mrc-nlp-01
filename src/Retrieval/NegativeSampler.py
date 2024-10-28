import numpy as np

class NegativeSampler:
    def __init__(self, corpus):
        self.corpus = np.array(corpus)
        
    def offer(self, row, num_negatives):
        context = row["context"]
        question = row["question"]
        
        while True:
            neg_idxs = np.random.randint(len(self.corpus), size=num_negatives)
            if not context in self.corpus[neg_idxs]:
                p_neg = self.corpus[neg_idxs]

                return {
                    "question": question,
                    "context": context,
                    "negatives": p_neg
                }

    def offer_bulk(self, dataset, num_negatives):
        questions = []
        contexts = []
        p_negs = []
        
        for row in dataset:
            sample = self.offer(row, num_negatives)
            questions.append(sample["question"])
            contexts.append(sample["context"])
            p_negs.append(sample["negatives"])
        
        return {
            "question": questions,
            "context": contexts,
            "negatives": p_negs
        }