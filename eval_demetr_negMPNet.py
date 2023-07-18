import argparse
import pandas as pd
import os
import numpy as np
import torch
import json
from sentence_transformers import SentenceTransformer, util

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--demetr_data_path", default="dataset/")
args = parser.parse_args()

base_model_name = "sentence-transformers/all-mpnet-base-v2"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
finetuned_model = SentenceTransformer("tum-nlp/NegMPNet", device=device)
base_model = SentenceTransformer(base_model_name, device=device)

def cos_score(reference: str, candidate: str, model:SentenceTransformer) -> float:
    emb_ref = model.encode(reference)
    emb_cand = model.encode(candidate)
    return util.cos_sim(emb_ref, emb_cand).item()

def cos_score_batched(references: list, candidates: list, model: SentenceTransformer, batch_size=8) -> torch.Tensor:
    assert len(references) == len(candidates), "Number of references and candidates must be equal"
    emb_ref = model.encode(references, batch_size=batch_size)
    emb_cand = model.encode(candidates, batch_size=batch_size)
    return torch.diag(util.cos_sim(emb_ref, emb_cand))

def demetr_accuracy_sent_transform(dataset: pd.DataFrame, model:SentenceTransformer, score_function) -> (float, np.array, np.array):
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
    print("** Base model")
    demetr_ratio(dataset, base_model, score_function)
    print("** Fine-tuned model")
    return demetr_ratio(dataset, finetuned_model, score_function)


def load_demetr_dataset(data_path:str) -> pd.DataFrame:
    df:pd.DataFrame = pd.read_json(args.demetr_data_path + data_path)
    return df
perturbation_datasets = {}
for filename in os.listdir(args.demetr_data_path):
    perturbation_datasets[filename.replace(".json", "")] = load_demetr_dataset(filename)

demetr_scores = {}
for pert_name, pert_data in perturbation_datasets.items():
    print("* ", pert_name.capitalize())
    dem_rat = eval_models_on_dataset(pert_data, cos_score_batched)
    demetr_scores[pert_name] = dem_rat
    print("\n")

json.dump(demetr_scores, open("demetr_scores_negMPNet.json", 'w+'))