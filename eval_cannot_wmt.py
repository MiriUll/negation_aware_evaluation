from tqdm import tqdm
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch

print("* Load CANNOT WMT test data")
cannot_wmt_test = pd.read_csv("cannot_wmt_data/cannot_wmt_test.csv")

print("* Load NegMPNnet ")
finetuned_model = SentenceTransformer("tum-nlp/NegMPNet")

def cos_similarities(references: list, candidates: list, model: SentenceTransformer, batch_size=8) -> torch.Tensor:
    assert len(references) == len(candidates), "Number of references and candidates must be equal"
    emb_ref = model.encode(references, batch_size=batch_size)
    emb_cand = model.encode(candidates, batch_size=batch_size)
    return torch.diag(util.cos_sim(emb_ref, emb_cand))

print("* Load NegBLEURT")
model_name = "tum-nlp/NegBLEURT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print("* Predicting NegMPNet scores")
scores_negmpnet = cos_similarities(cannot_wmt_test.reference, cannot_wmt_test.candidate, finetuned_model)
print("* NegMPNet results:")
print("** Pearson:", pearsonr(scores_negmpnet, cannot_wmt_test.score).statistic)
print("** Spearman:", spearmanr(scores_negmpnet.tolist(), cannot_wmt_test.score).statistic)

print("* Predicting NegBLEURT scores")
references = list(cannot_wmt_test.reference.values)
candidates = list(cannot_wmt_test.candidate.values)
scores_negbleut = []
batch_size = 16
for i in tqdm(range(0, len(references), batch_size)):
  tokenized = tokenizer(references[i:i+batch_size], candidates[i:i+batch_size], return_tensors='pt', padding=True, max_length=512, truncation=True)
  scores_negbleut += model(**tokenized).logits.flatten().tolist()
print("* NegBLEURT results:")
print("** Pearson:", pearsonr(scores_negbleut, cannot_wmt_test.score).statistic)
print("** Spearman:", spearmanr(scores_negbleut, cannot_wmt_test.score).statistic)
