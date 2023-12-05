# ⚙️ Invalidator ✂️
*by Thanh Le-Cong, Duc-Minh Luong, Xuan-Bach D. Le, David Lo, Nhat-Hoa Tran, Quang-Huy Bui, Quyet Thang Huynh*

This repository contains source code of research paper "Invalidator: Automated Patch Correctness Assessment via Semantic and Syntactic Reasoning", which is published IEEE Transactions on Software Engineering.

<p align="center">
    <a href="https://ieeexplore.ieee.org/document/10066209"><img src="https://img.shields.io/badge/Journal-IEEE TSE Volume 49 (2023)-green?style=for-the-badge">
    <a href="https://arxiv.org/pdf/2301.01113.pdf"><img src="https://img.shields.io/badge/arXiv-2301.01113-b31b1b.svg?style=for-the-badge">
</p>

## 💥 Approach

Automated program repair (APR) faces the challenge of test overfitting, where generated patches pass validation tests but fail to generalize. Existing methods for patch assessment involve generating new tests or manual inspection, which can be time-consuming or biased. In this paper, we propose a novel technique, Invalidator, to automatically assess the correctness of APR-generated patches via semantic and syntactic reasoning. Invalidator leverages program invariants to reason about program semantics while also capturing program syntax through language semantics learned from a large code corpus using a Large Language Model. 

Given a buggy program and the developer-patched program, Invalidator infers likely invariants on both programs. Then, Invalidator determines that an APR-generated patch overfits if: (1) it violates correct specifications or (2) maintains erroneous behaviors from the original buggy program. In case our approach fails to determine an overfitting patch based on invariants, Invalidator utilizes a trained model from labeled patches to assess patch correctness based on program syntax. The benefit of Invalidator is threefold. First, Invalidator leverages both semantic and syntactic reasoning to enhance its discriminative capability. Second, Invalidator does not require new test cases to be generated, but instead only relies on the current test suite and uses invariant inference to generalize program behaviors. Third, Invalidator is fully automated. 

<p align="center">
<img width="850" alt="Screenshot 2023-12-05 at 9 52 28 pm" src="https://github.com/thanhlecongg/Invalidator/assets/43113794/b6d337d6-168c-4105-b806-6e4328cd981e">
</p>


## 📈 Experimental Results

Experimental results demonstrate that Invalidator outperforms existing methods in terms of Accuracy and F-measure, correctly identifying 79% of overfitting patches and detecting 23% more overfitting patches than the best baseline.

<p align="center">
<img width="650" alt="Screenshot 2023-12-05 at 9 52 40 pm" src="https://github.com/thanhlecongg/Invalidator/assets/43113794/f0b8325e-7886-4fd5-a570-70d4814da95b">
</p>
