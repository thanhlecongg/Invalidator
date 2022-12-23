# Invalidator

This repository contains source code of research paper "Invalidator: Automated Patch Correctness Assessment via Semantic and Syntactic Reasoning"

## Structure
The structure of our source code's repository is as follows:
- data: contains our experimental data;
- model: contains the trained model for syntactic-based classifier on our experiments;
- src: contains our source code.
    - classifier: contain source code for our classifiers including: semantic and syntactic classifiers
    - feature: contains  source code for syntactic feature (CodeBERT, BERT, ODS) preparation 
    - utils.py: contains source code for utility functions, e.g., logger, visualization, ...
- environment.yml: contains the configuration for AutoPruner's enviroment. 

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
