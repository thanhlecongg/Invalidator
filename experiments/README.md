# Experiments
This README contains detailed instruction to replicate our experimental results on research paper "Invalidator: Automated Patch Correctness Assessment via Semantic and Syntactic Reasoning"

Before replicating our experimental results, please  unzip our .zip file in folder data.

### Invalidator
To replicate the results of Invalidator, please use the following commands:
```
python3 invalidator.py
```
To replicate the results of Invalidator's Semantic Classifier, please use the following commands:
```
python3 invalidator.py --c 0
```
To replicate the results of Invalidator's Syntactic Classifier, please use the following commands:
```
python3 invalidator.py --c 1
```

### Syntactic Classifier

#### With grountruth
To replicate the results of Invalidator's Syntactic Classifier with CodeBERT features, please use the following commands:
```
python3 -m src.classifier.syntactic_classifier
```
To replicate the results of Invalidator's Syntactic Classifier with BERT features, please use the following commands:
```
python3 -m src.classifier.syntactic_classifier_BERT
```
To replicate the results of Invalidator's Syntactic Classifier with ODS features, please use the following commands:
```
python3 -m src.classifier.syntactic_classifier_ods
```

#### W/o grountruth
To replicate the results of Invalidator's Syntactic Classifier with CodeBERT features, please use the following commands:
```
python3 -m src.classifier.syntactic_classifier_wo_gt
```
To replicate the results of Invalidator's Syntactic Classifier with BERT features, please use the following commands:
```
python3 -m src.classifier.syntactic_classifier_BERT_wo_gt
```
To replicate the results of Invalidator's Syntactic Classifier with ODS features, please use the following commands:
```
python3 -m src.classifier.syntactic_classifier_ods_wo_gt
```