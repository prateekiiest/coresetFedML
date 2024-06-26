# Bayesian Coreset Optimization for Personalized Federated Learning
### _Prateek Chanda, Shrey Modi, Ganesh Ramakrishnan_

### Accepted at International Conference on Learning Representations (ICLR) 2024


[Project Page](https://coresetfederatedlearning.github.io/) | [OpenReview](https://openreview.net/forum?id=uz7d2N2zul) | [Paper](https://openreview.net/pdf?id=uz7d2N2zul) | [Poster](https://iclr.cc/media/PosterPDFs/ICLR%202024/17557.png?t=1713590458.7719705) | [Slides](https://iclr.cc/media/iclr-2024/Slides/17557.pdf) |

------------------------------

### Abstract
In a distributed machine learning setting like Federated Learning where there are multiple clients involved which update their individual weights to a single central server, often training on the entire individual client's dataset for each client becomes cumbersome. To address this issue we propose CORESETPFEDBAYES: a personalized coreset weighted federated learning setup where the training updates for each individual clients are forwarded to the central server based on only individual client coreset based representative data points instead of the entire client data. Through theoretical analysis we present how the average generalization error is minimax optimal up to logarithm bounds - upper bounded by $\mathcal{O}(n_k^{-\frac{2 \beta}{2 \beta+\boldsymbol{\Lambda}}} \log ^{2 \delta^{\prime}}(n_k))$ and lower bounds of $\mathcal{O}(n_k^{-\frac{2 \beta}{2 \beta+\boldsymbol{\Lambda}}})$, and how the overall generalization error on the data likelihood differs from a vanilla Federated Learning setup as a closed form function ${\boldsymbol{\Im}}(\boldsymbol{w}, n_k)$ of the coreset weights $\boldsymbol{w}$ and coreset sample size $n_k$. 


### Citation
```
@inproceedings{
      chanda2024bayesian,
      title={Bayesian Coreset Optimization for Personalized Federated Learning},
      author={Prateek Chanda and Shrey Modi and Ganesh Ramakrishnan},
      booktitle={The Twelfth International Conference on Learning Representations},
      year={2024},
      url={https://openreview.net/forum?id=uz7d2N2zul}
      }
```

