import pandas as pd
import os
import numpy as np
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--demetr_data_path", default="demetr/dataset/")
args = parser.parse_args()
demetr_data_path = args.demetr_data_path

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
tokenizer = AutoTokenizer.from_pretrained("tum-nlp/NegBLEURT")
model = AutoModelForSequenceClassification.from_pretrained("tum-nlp/NegBLEURT").to(device=device)

def score(ref, cand, model):
    if type(ref) != list:
        ref=ref.values.tolist()
    if type(cand) != list:
        cand=cand.values.tolist()
    tokenized = tokenizer(ref, cand, return_tensors='pt', padding=True)
    return model(input_ids=tokenized['input_ids'].to(device), token_type_ids=tokenized['token_type_ids'].to(device), attention_mask=tokenized['attention_mask'].to(device)).logits.flatten()

def demetr_accuracy_sent_transform(dataset: pd.DataFrame, model, score_function) -> (float, np.array, np.array):
    t_scores = score_function(dataset.eng_sent, dataset.mt_sent, model)
    hat_scores = score_function(dataset.eng_sent, dataset.pert_sent, model)
    return sum(torch.greater(t_scores, hat_scores)) / len(dataset), t_scores, hat_scores

def demetr_ratio(dataset: pd.DataFrame, model, score_function) -> float:
    acc, t_scores, hat_scores = demetr_accuracy_sent_transform(dataset, model, score_function)
    print(f"Detection accuracy: {acc}")
    empty_scores = score_function(dataset.eng_sent, ["."] * len(dataset), model)
    ratio = (t_scores - hat_scores) / (t_scores - empty_scores)
    ratio = sum(ratio) / len(dataset)
    print(f"Ratio: {ratio}")
    return ratio.item()

def eval_models_on_dataset(dataset:pd.DataFrame, score_function) -> None:
    print("** Fine-tuned model")
    return demetr_ratio(dataset, model, score_function)


def load_demetr_dataset(data_path:str) -> pd.DataFrame:
    df:pd.DataFrame = pd.read_json(demetr_data_path + data_path)
    return df
perturbation_datasets = {}
for filename in os.listdir(demetr_data_path):
    perturbation_datasets[filename.replace(".json", "")] = load_demetr_dataset(filename)

demetr_scores = {}
for pert_name, pert_data in perturbation_datasets.items():
    print("* ", pert_name.capitalize())
    dem_rat = eval_models_on_dataset(pert_data, score)
    demetr_scores[pert_name] = dem_rat
    print("\n")

json.dump(demetr_scores, open("demetr_scores_negBleurt.json", 'w+'))