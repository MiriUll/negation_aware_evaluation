# Negation-aware evaluation of Seq2seq systems
This repository contains the code for the paper "This is not correct! Negation-aware Evaluation of Language Generation Systems".  
Our pre-trained models are available on Huggingface. The out-of-the-box usage is indicated on their respective model pages:  
- [NegMPNet](https://huggingface.co/tum-nlp/NegMPNet), a negation-aware sentence transformer
- [NegBLEURT](https://huggingface.co/tum-nlp/NegBLEURT), a negation-aware metric for reference-based evaluation of NLG systems

## Re-creating fine-tuned models
### NegMPNet
The NegMPNet model was trained using the [finetune_MPNet](https://github.com/MiriUll/negation_aware_evaluation/blob/master/finetune_MPNet.py) script, adapted from the [NLI training script](https://github.com/UKPLab/sentence-transformers/blob/3e1929fddef16df94f8bc6e3b10598a98f46e62d/examples/training/nli/training_nli_v2.py) from the official sentence-transformers documentation.
We used a MultipleNegativesRankingLoss. For hyperparameter settings, please check our paper or look at the code. 
You can run the script after installing the system requirements with ```python finetune_MPNet.py```. This will create a folder ```finetuned-models``` in which you can find the fine-tuned model.

### NegBLEURT
We fine-tuned BLEURT using the codebase of the official [BLEURT repo](https://github.com/google-research/bleurt). To re-create the model, run the following commands:  
```
git clone https://github.com/google-research/bleurt:bleurtMaster
cd bleurtMaster
```
Then, recreate our NegBLEURT checkpoint with this command:
```
python -m bleurt.finetune -init_bleurt_checkpoint=bleurt\test_checkpoint \ 
-model_dir=..\finetuned-models\neg_bleurt_500 \
-train_set=..\cannot_wmt_data\cannot_wmt_train.jsonl \
-dev_set=..\cannot_wmt_data\cannot_wmt_eval.jsonl \
-num_train_steps=500
```
This will store the fine-tuned model into the ```neg_bleurt_500/export/bleurt_best/{id}/``` folder where id is some random timestamp. We recommend to copy the files in this folder into the ```neg_bleurt_500``` to avoid having to lookup the exact ID.  
Afterwards, we converted the fine-tuned model to a PyTorch transformer model using [this Github repository](https://github.com/lucadiliello/bleurt-pytorch). Our adapted code for converting our model is published in [this notebook](https://github.com/MiriUll/negation_aware_evaluation/blob/master/development-notebooks/Bleurt_to_transformers.ipynb).

## Re-creating values from the paper
We evaluated our models un multiple benchmark datasets.
### CANNOT-WMT test data
Our CANNOT-WMT data has a held-out test set to examine negation understanding of evaluation metrics. To evaluate our models on this dataset, run ```python eval_cannot_wmt.py```.  
Please refer to the [CANNOT Github repository](https://github.com/dmlls/cannot-dataset/) for further details about the dataset and its creation.
### Massive Text Embedding benchmark (MTEB)
To evaluate NegMPNet on MTEB, first you need to install [mteb](https://github.com/embeddings-benchmark/mteb):
```
pip install mteb
```
Then, use the following code to run NegMPNet on the benchmark.
```
from mteb import MTEB
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
model_name = "tum-nlp/NegMPNet"

model = SentenceTransformer(model_name)
evaluation = MTEB(task_langs=["en"])
results = evaluation.run(model, output_folder="mteb_results")
```
For us, some dataset returned error or could not be calculated due to hardware limitations. Look at the official repository for more detailed task specification if some tasks don't work. 
### Metrics Comparison benchmark
We used the [original Github repository](https://github.com/LazerLambda/MetricsComparison) to evaluate our models and their baselines on this benchmark. For the BLEURT-based models, we used the original [BLEURTRec](https://github.com/LazerLambda/MetricsComparison/blob/master/src/metrics/custom_metrics/BLEURTRec.py) implementation and updated the paths to our models. For NegMPNet, we created ```SBERTMetric.py```. Copy this file to the ```custom_metrics``` folder of the Metrics Comparison repo to include NegMPNet in the evaluation.
### DEMETR
Download the [DEMETR repository](https://github.com/marzenakrp/demetr) into the folder ```demetr```.  
To evaluate NegBLEURT on the DEMETR benchmark, run 
```
 python eval_demetr_negBleurt.py -d path_to_demetr_dataset
```
This will store the scores in ```demetr_scores_negBleurt.json```.  
For evaluating NegMPNET, run 
```
python eval_demetr_negMPNet.py -d path_to_demetr_dataset
 ```
This will print the score of the baseline model as well but will only store the NegMPNet results into ```demetr_scores_negMPNet.json```.
## Citation
Please cite our INLG 2023 paper, if you use our repository, models or data:  
```bibtex
@misc{anschütz2023correct,
      title={This is not correct! Negation-aware Evaluation of Language Generation Systems}, 
      author={Miriam Anschütz and Diego Miguel Lozano and Georg Groh},
      year={2023},
      eprint={2307.13989},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
