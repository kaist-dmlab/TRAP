# TRAP

## TRAP: Two-level Regularized Autoencoder-based Embedding for Power-law Distributed Data

**TRAP** is a general and powerful regularizer for autoencoder (AE)-based embedding methods on the graph data where the data follow the power-law distribution w.r.t the sparsity of input vectors. **TRAP** significantly boosts performances of two represensitive graph embedding tasks, *(1) Top-k recommendation* on user-item transaction datasets and *(2) Node classification* on common graph datasets, by up to *31.53%* and *94.99%* respectively.

## Citations

```
@inproceedings{park2020trap,
  title={TRAP: Two-level regularized autoencoder-based embedding for power-law distributed data},
  author={Park, Dongmin and Song, Hwanjun and Kim, Minseok and Lee, Jae-Gil},
  booktitle={Proceedings of The Web Conference 2020},
  pages={1615--1624},
  year={2020}
}
```


## Setup

- python3
- tensorflow v1.14.0

## Usage

Two tasks are used to evaluate the effectiveness of **TRAP**; *(1)Top-k recommendation with user-item embedding* and *(2) Node classification with node embedding*. The source code of each task is in the *UserEmbedding* and *NodeEmbedding* folders, respectively. Detail instructions of how to run the codes can be found on Readme.md files in each folder.

## Code References

Because **TRAP** is a meta-algorithm (regularizer), we combined it with existing baseline methods to improve the performance. The github baseline codes we mainly used are the following:

User-item Embedding
- https://github.com/Zziwei/Joint-Collaborative-Autoencoder, Joint Collaborative Autoencoder (JCA), TheWebConf, 2019. 

Node Embedding
- https://github.com/suanrong/SDNE, Structural Deep Network embedding (SDNE), SIGKDD, 2016.

Thank you for the authors.

## Contact

Please post a Github issue or contact dongminpark@kaist.ac.kr if you have any questions. Thanks.
