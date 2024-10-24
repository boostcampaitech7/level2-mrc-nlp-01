import os
import json
import collections
import numpy as np
from ensemble_utils import normalize_answer
from datasets import load_metric, load_from_disk

def compute_metric(pred_dict):
    dataset = load_from_disk("data/train_dataset")
    metric = load_metric("squad")
    
    valid_dataset = dataset["validation"]
    predictions = [{"id": k, "prediction_text" : v} for k, v in pred_dict.items() if k in valid_dataset["id"] ]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in valid_dataset]
    
    if len(predictions) != len(references):
        return None
    
    results = metric.compute(predictions=predictions, references=references)
    return results

def do_ensemble(dirname, weights, file_names):
    predictions = collections.defaultdict(list)
    
    for file_name in file_names:
        with open(os.path.join(dirname, file_name), 'r', encoding='utf-8') as f:
            data = json.load(f)
            for id, preds in data.items():
                predictions[id].append(preds)
    
    for id, preds in predictions.items():
        if len(preds) != len(weights):
            print(f"! ! ! Warning: '{id}' has {len(preds)} predictions, but {len(weights)} weights are given. ! ! !")
    
    id_to_prob = collections.defaultdict(
        lambda: collections.defaultdict(lambda: tuple([list(), float()]))
    )
    
    for id, preds in predictions.items():
        for i, pred in enumerate(preds):
            for each in pred:
                normalized = normalize_answer(each["text"])
                pretexts, prob = id_to_prob[id][normalized]
                pretexts.append(each["text"])
                prob += (weights[i] / sum(weights)) * each["probability"]
                id_to_prob[id][normalized] = (pretexts, prob)
    
    ensembled = {}
    for id, text_to_prob in id_to_prob.items():
        probs = []
        pretexts = []
        texts = []
        for text, (pretext, prob) in text_to_prob.items():
            texts.append(text)
            probs.append(prob)
            pretexts.append(pretext)
        max_idx = np.argmax(probs)
        ensembled[id] = pretexts[max_idx][0]
    
    with open('ensemble.json', 'w', encoding='UTF-8') as f:
        json.dump(ensembled, f, indent=4, ensure_ascii=False)
    
    results = compute_metric(ensembled)
    print(results)

def main():
    dirname = input(
        "Enter the directory name where the prediction files are located (default 'output/ensemble'): "
    )
    if dirname == "":
        dirname = "outputs/ensemble"
        
    file_names = os.listdir(dirname)
    file_names = [a for a in file_names if a[-5:] == ".json"]
    weights = [0.0] * len(file_names)
    
    page = 1
    while True:
        print(f"Page {page}")
        for i, file_name in enumerate(file_names[(page - 1) * 10: page * 10]):
            weight = weights[(page - 1) * 10 + i]
            print(f"{'v' if weight != 0 else ' '} {i}. {file_name} ", end="") 
            print(f"({weight})" if weight != 0 else "")
    
        choice = input("Enter the number of the file you want to check [0-9,a,s,d]: ")
        if choice == 's':
            break
        elif choice == 'a':
            page = max(1, page - 1)
            continue
        elif choice == 'd':
            page = min(len(file_names) // 10 + 1, page + 1)
            continue
        
        try:
            choice = int(choice)
            if choice < 0 or choice > 9:
                raise ValueError
        except ValueError:
            print("Please enter a number between 0 and 9.")
            continue
        
        select = (page - 1) * 10 + choice
        
        weight = input(f"Enter the weight of '{file_names[select]}' (default = 1.0): ")
        weight = float(weight) if weight != "" else 1.0
        weights[select] = weight
    
    print("Let's cook!")
    do_ensemble(dirname, weights, file_names)
        
if __name__ == '__main__':
    main()