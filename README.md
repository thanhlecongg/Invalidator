# Invalidator

This repository contains source code of research paper "Invalidator: Automated Patch Correctness Assessment via Semantic and Syntactic Reasoning"

## Structure

```
├── data
│   ├── processed_data
│   │   ├── b_invariants [containing invariant inferred by Daikon on buggy methods]
│   │   ├── invariants [containing invariant inferred by Daikon on executed methods]
│   │   └── *.pkl [containing features extracted from CodeBERT/BERT/ODS]
│   ├── raw_data
│   │   ├── ASE20_Patches [containing patches from Wang et al. "Automated Patch Correctness Assessment: How Far are We?" (ASE 2020)]
│   │   ├── defects4j-developer [containing Defects4j developer patches]
│   │   ├── ICSE18_Patches [containing patches from Xiong et al. "Identifying Patch Correctness in Test-Based Program Repair?" (ICSE 2018)]
│   │   ├── ODS_features [containing extracted features from Ye et al. "Automated Classification of Overfitting Patches with Statically Extracted Code Features"]
│   │   └── patch_info [containing information of patches in our dataset]
├── environment.yml [the configuration for Invalidator's enviroment]
├── experiments
│   ├── invalidator_log [containing logs/results produced by Invalidator]
│   ├── syntactic_classifier_log [containing logs/results produced by Synatactic-based Classifiers]
│   └── README.md [containing instructions to replicate our results]
├── model [containing trained syntactic classifiers]
├── README.md [containing instructions to use Invaliator]
├── invalidator.py [containing the main file of Invalidator]
└── src
    ├── classifier [containing source code for our classifier]
    ├── feature [containing source code for our feature extractor]
    └── utils.py [containing source code for utility functions, e.g., logger, visualization, ...]
```

## Environment Configuration
### Conda
```
conda env create -f environment.yml
```

- Activate conda:
```
source /opt/conda/bin/activate
```
- Activate Invalidator's conda enviroment: 
```
conda activate invalidator
```

## Usage
To use our tool, please use the following command
```
python3 -m invalidator.py --c [classifier: 0: semantic, 1: syntactic, 2:combine]
                          --T [threshold of syntactic classifier] 
```
- Option "--c": Invalidator provide three option for classifier including:
    - semantic: With this `semantic` options, Invalidator use semantic classifier which assess the correctness of APR-generated patches via program invariants inferred from Daikon and two proposed overfitting rules (See Section 4.1 in our paper);
    - syntactic: With this `syntatic` options, Invalidator use syntatic classifier which assess the correctness of APR-generated patches via CodeBERT representation and a Logistic Regression classifier (See Section 4.2 in our paper);
    - combine: This is a default option, which use both semantic and syntactic classifiers to assess the correctness of APR-generated patches. 
    
 - Option "--T": This option define the classification threshold for our syntactic classifier
    - By default, this option is set as 0.975;
    - Users can change this threshold depeding on their purpose.
