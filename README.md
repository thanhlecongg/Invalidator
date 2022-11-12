# AutoPruner

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
