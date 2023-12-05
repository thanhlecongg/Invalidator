# ⚙️ Invalidator ✂️
*by Thanh Le-Cong, Duc-Minh Luong, Xuan-Bach D. Le, David Lo, Nhat-Hoa Tran, Quang-Huy Bui, Quyet Thang Huynh*

This repository contains source code of research paper "Invalidator: Automated Patch Correctness Assessment via Semantic and Syntactic Reasoning", which is published IEEE Transactions on Software Engineering.

<p align="center">
    <a href="https://ieeexplore.ieee.org/document/10066209"><img src="https://img.shields.io/badge/Journal-IEEE TSE Volume 49 (2023)-green?style=for-the-badge">
    <a href="https://arxiv.org/pdf/2301.01113.pdf"><img src="https://img.shields.io/badge/arXiv-2301.01113-b31b1b.svg?style=for-the-badge">
    <br>
</p>

## 📃 Overview
If you are interested in our work, please refer to our [overview](https://github.com/thanhlecongg/Invalidator/overview.md) for more details.
      
## 🏁 Repository Organization

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

## 🔧 Installations
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

## 🚀 Usage
To use our tool, please use the following command
```
python3 experiment.py --c [classifier: 0: semantic, 1: syntactic, 2:combine]
                          --T [threshold of syntactic classifier] 
```
- Option "--c": Invalidator provide three option for classifier including:
    - semantic: With this `semantic` options, Invalidator use semantic classifier which assess the correctness of APR-generated patches via program invariants inferred from Daikon and two proposed overfitting rules (See Section 4.1 in our paper);
    - syntactic: With this `syntatic` options, Invalidator use syntatic classifier which assess the correctness of APR-generated patches via CodeBERT representation and a Logistic Regression classifier (See Section 4.2 in our paper);
    - combine: This is a default option, which use both semantic and syntactic classifiers to assess the correctness of APR-generated patches. 
    
 - Option "--T": This option define the classification threshold for our syntactic classifier
    - By default, this option is set as 0.975;
    - Users can change this threshold depeding on their purpose.

## 📜 Citation
Please cite the following article if you find Invalidator to be useful:

```
@article{le2023invalidator,
  author={Le-Cong, Thanh and Luong, Duc-Minh and Le, Xuan Bach D. and Lo, David and Tran, Nhat-Hoa and Quang-Huy, Bui and Huynh, Quyet-Thang},
  journal={IEEE Transactions on Software Engineering}, 
  title={Invalidator: Automated Patch Correctness Assessment Via Semantic and Syntactic Reasoning}, 
  year={2023},
  volume={49},
  number={6},
  pages={3411-3429},
  doi={10.1109/TSE.2023.3255177}
}
```
