import logging
import math
import os
import time
import csv
import random
import torch
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch import nn, Tensor
from typing import Iterable, Dict
import pandas as pd


negation_dataset_path = "cannot_wmt_data/"
base_model = "sentence-transformers/all-mpnet-base-v2"
output_model_name = f"{base_model.split('/')[1]}-negation-wmt"
model_save_path = str(f"finetuned-models/{output_model_name}")
if not os.path.exists("finetuned-models"):
    os.mkdir("finetuned-models")

train_batch_size = 64  # The larger you select this, the better the results (usually). But it requires more GPU memory
max_seq_length = 75
num_epochs = 1

class MultipleNegativesRankingLoss(nn.Module):
    """
        This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
        where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.

        For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
        n-1 negative examples (p_j). It then minimizes the negative log-likehood for softmax normalized scores.

        This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc))
        as it will sample in each batch n-1 negative docs randomly.

        The performance usually increases with increasing batch sizes.

        For more information, see: https://arxiv.org/pdf/1705.00652.pdf
        (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)

        You can also provide one or multiple hard negatives per anchor-positive pair by structering the data like this:
        (a_1, p_1, n_1), (a_2, p_2, n_2)

        Here, n_1 is a hard negative for (a_1, p_1). The loss will use for the pair (a_i, p_i) all p_j (j!=i) and all n_j as negatives.

        Example::

            from sentence_transformers import SentenceTransformer, losses, InputExample
            from torch.utils.data import DataLoader

            model = SentenceTransformer('distilbert-base-uncased')
            train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
                InputExample(texts=['Anchor 2', 'Positive 2'])]
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
            train_loss = losses.MultipleNegativesRankingLoss(model=model)
    """
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct = util.cos_sim):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(MultipleNegativesRankingLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        print(similarity_fct)


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        return self.cross_entropy_loss(scores, labels)

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Here we define our SentenceTransformer model
word_embedding_model = models.Transformer(base_model, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


def load_data(dataset_path):
    def add_to_samples(sent1, sent2, label):
        if sent1 not in train_data:
            train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
        #label = "contradiction" if label == "1" else "entailment"
        label = "contradiction" if float(label) <= 0.5 else "entailment"
        train_data[sent1][label].add(sent2)

    train_data = {}
    neg_data = pd.read_csv(dataset_path)
    print("Number of samples:", len(neg_data), "\n")
    for i, row in neg_data.iterrows():
        sent1 = row['reference'].strip()
        sent2 = row['candidate'].strip()
        add_to_samples(sent1, sent2, row['score'])

    train_samples = []
    for sent1, others in train_data.items():
        if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
            train_samples.append(InputExample(texts=[sent1, random.choice(list(others['entailment'])), random.choice(list(others['contradiction']))]))
            train_samples.append(InputExample(texts=[random.choice(list(others['entailment'])), sent1, random.choice(list(others['contradiction']))]))
    return train_samples


# Read the negation dataset file and create the training dataset
logging.info("Read Negation Dataset train dataset")
negation_dataset = negation_dataset_path + "cannot_wmt_train.csv"
train_samples=load_data(negation_dataset)
# Special data loader that avoid duplicates within a batch
train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)


# Our training loss
train_loss = MultipleNegativesRankingLoss(model)

dev_samples = load_data(negation_dataset_path + "cannot_wmt_eval.csv")
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='wmt-dev')

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
eval_log_steps = int(len(train_dataloader)*0.1)
start_time = time.perf_counter()
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=num_epochs,
    evaluation_steps=eval_log_steps,
    warmup_steps=warmup_steps,
    output_path=model_save_path,
    use_amp=True  # Set to True, if your GPU supports FP16 operations
)
elapsed_time = time.perf_counter() - start_time

print("\n\nTraining time (seconds):", elapsed_time)

model.save(model_save_path)
print("\n\nModel saved")

base_mod = SentenceTransformer(base_model)
model.eval()
def cos_score(reference: str, candidate: str, model:SentenceTransformer) -> float:
    emb_ref = model.encode(reference)
    emb_cand = model.encode(candidate)
    return util.cos_sim(emb_ref, emb_cand).item()

def cos_score_batched(references: list, candidates: list, model: SentenceTransformer, batch_size=8) -> torch.Tensor:
    assert len(references) == len(candidates), "Number of references and candidates must be equal"
    emb_ref = model.encode(references, batch_size=batch_size)
    emb_cand = model.encode(candidates, batch_size=batch_size)
    return torch.diag(util.cos_sim(emb_ref, emb_cand))

sent_pairs = [
    ("It's rather hot in here.", "It's rather cold in here."),
    ("This is a red cat with a hat.", "This isn't a red cat with a hat."),
    ("This is a red cat with a hat.", "This is not a red cat with a hat."),
    ("Today is a beautiful day.", "Today is a wonderful day."),
    ("Today is a beautiful day.", "beautiful day today is."),
    ("Today is a beautiful day.", "today today today today is a beautiful day."),
    ("Today is a beautiful day.", "Today is a betiful day."),
    ("Today is a beautiful day.", "Today is a really beautiful day."),
    ("Today is a beautiful day.", "Today is a beautiful day."),
    ("Today is a beautiful day.", "."),
]

for s1, s2 in sent_pairs:
    print(s1)
    print(s2)
    print("Base", cos_score_batched([s1], [s2], base_mod))
    print("FT", cos_score_batched([s1], [s2], model))