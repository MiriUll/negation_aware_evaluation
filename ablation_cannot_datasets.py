from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import torch
import json
from bleurtMaster.bleurt.score import BleurtScorer
import sys
sys.path.append("bleurtMaster")

bleurt_full = BleurtScorer(checkpoint="neg_bleurt_500")
bleurt_wiki = BleurtScorer(checkpoint="finetuned-models/neg_bleurt_no_wiki/export/bleurt_best/1689775316")
bleurt_nan_nli = BleurtScorer(checkpoint="finetuned-models/neg_bleurt_no_nan_nli/export/bleurt_best/1689776327")
bleurt_sentiment = BleurtScorer(checkpoint="finetuned-models/neg_bleurt_no_sentiment/export/bleurt_best/1689831042")
bleurt_glue = BleurtScorer(checkpoint="finetuned-models/neg_bleurt_no_glue/export/bleurt_best/1689832466")

"""
print("* Load CANNOT WMT test data")
cannot_wmt_test = pd.read_csv("cannot_wmt_data/cannot_wmt_test.csv")

scores = {
    'full': [],
    'wiki': [],
    'nan_nli': [],
    'sentiment': [],
    'glue': [],
}
for _, row in tqdm(cannot_wmt_test.iterrows(), total=len(cannot_wmt_test)):
    scores['full'] += bleurt_full.score(references=[row.reference], candidates=[row.candidate])
    scores['wiki'] += bleurt_wiki.score(references=[row.reference], candidates=[row.candidate])
    scores['nan_nli'] += bleurt_nan_nli.score(references=[row.reference], candidates=[row.candidate])
    scores['sentiment']+= bleurt_sentiment.score(references=[row.reference], candidates=[row.candidate])
    scores['glue'] += bleurt_glue.score(references=[row.reference], candidates=[row.candidate])

for model_name, preds in scores.items():
    print("*", model_name)
    print("**", pearsonr(preds, cannot_wmt_test.score).statistic)
    print("**", spearmanr(preds, cannot_wmt_test.score).statistic)

print("\n\n")"""

def demetr_accuracy_bleurt(dataset: pd.DataFrame, bleurt_scorer:BleurtScorer) -> (float, np.array, np.array):
    t_scores = torch.tensor(bleurt_scorer.score(references=dataset.eng_sent, candidates=dataset.mt_sent))
    hat_scores = torch.tensor(bleurt_scorer.score(references=dataset.eng_sent, candidates=dataset.pert_sent))
    return sum(torch.greater(t_scores, hat_scores)) / len(dataset), t_scores, hat_scores


def demetr_ratio_bleurt(dataset: pd.DataFrame, bleurt_scorer:BleurtScorer) -> float:
    acc, t_scores, hat_scores = demetr_accuracy_bleurt(dataset, bleurt_scorer)
    #print(f"Detection accuracy: {acc}")
    empty_scores = torch.tensor(bleurt_scorer.score(references=dataset.eng_sent, candidates=["."] * len(dataset)))
    ratio = (t_scores - hat_scores) / (t_scores - empty_scores)
    ratio = sum(ratio) / len(dataset)
    print(f"Ratio: {ratio}")
    return ratio.item()

def eval_models_on_dataset_bleurt(dataset: pd.DataFrame):
    scores = {}
    print("** Full FT model")
    scores['full'] = demetr_ratio_bleurt(dataset, bleurt_full)
    print("** Wiki model")
    scores['wiki'] = demetr_ratio_bleurt(dataset, bleurt_wiki)
    print("** Nan nli model")
    scores['nan_nli'] = demetr_ratio_bleurt(dataset, bleurt_nan_nli)
    print("** Sentiment model")
    scores['sentiment'] = demetr_ratio_bleurt(dataset, bleurt_sentiment)
    print("** Glue model")
    scores['glue'] = demetr_ratio_bleurt(dataset, bleurt_glue)
    return scores

def load_demetr_dataset(data_path:str) -> pd.DataFrame:
    df:pd.DataFrame = pd.read_json(demetr_data_path + data_path)
    return df

demetr_data_path = "demetrMain/"
perturbation_datasets = {}
for filename in os.listdir(demetr_data_path):
    perturbation_datasets[filename.replace(".json", "")] = load_demetr_dataset(filename)

demetr_scores = {}
for pert_name, pert_data in perturbation_datasets.items():
    print("* ", pert_name.capitalize())
    demetr_scores[pert_name] = eval_models_on_dataset_bleurt(pert_data)
    json.dump(demetr_scores, open("cannot_ablation_demetr.json", 'w+'))
    print("\n")
