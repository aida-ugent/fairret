# fairret - a fairness library in PyTorch

[![Licence](https://img.shields.io/github/license/aida-ugent/fairret)](https://github.com/aida-ugent/fairret/blob/main/LICENSE)
[![PyPI - Version](https://img.shields.io/pypi/v/fairret)](https://pypi.org/project/fairret/)
![Static Badge](https://img.shields.io/badge/PyTorch-ee4c2c)
[![Static Badge](https://img.shields.io/badge/Original%20Paper-00a0ff)](https://openreview.net/pdf?id=NnyD0Rjx2B)

## Description

The goal of fairret is to serve as an open-source Python library for measuring and mitigating statistical fairness in PyTorch models. The library is designed to be 
1. *flexible* in how fairness is defined and pursued.
2. *easy* to integrate into existing PyTorch pipelines.
3. *clear* in what its tools can and cannot do.

The central to the library is the paradigm of the _fairness regularization term_ (fairrets) that quantify unfairness as differentiable PyTorch loss functions. 
These can then be optimized together with e.g. the binary cross-entropy error such that the classifier improves both its accuracy and fairness.

**The library is still in very early development.** Documentation, installation instructions, and more examples will be added in the near future.

## Installation
The fairret library can be installed via PyPi:

```
pip install fairret
```

### Dependencies
A minimal list of dependencies is provided in [pyproject.toml](https://github.com/aida-ugent/fairret/blob/main/pyproject.toml). If the library is installed locally, the required packages can be installed via `pip install .`

## Quickstart

It suffices to simply choose a statistic that should be equalized across groups and a fairret that quantifies the gap. The model can then be trained as follows:

```python
import torch.nn.functional as F
from fairret.statistic import PositiveRate
from fairret.loss import NormLoss

statistic = PositiveRate()
norm_fairret = NormLoss(statistic)

def train(model, optimizer, train_loader):
     for feat, sens, target in train_loader:
            optimizer.zero_grad()
            
            logit = model(feat)
            bce_loss = F.binary_cross_entropy_with_logits(logit, target)
            fairret_loss = norm_fairret(logit, sens)
            loss = bce_loss + fairret_loss
            loss.backward()
            
            optimizer.step()
```

No special data structure is required for the sensitive features. If the training batch contains N elements, then `sens` should be a tensor of floats with shape (N, d_s), with d_s the number of sensitive features. **Like any categorical feature, it is expected that categorical sensitive features are one-hot encoded.**

A notebook with a full example pipeline is provided here: [simple_pipeline.ipynb](/examples/simple_pipeline.ipynb).

## Warning: AI fairness != fairness
There are many ways in which technical approaches to AI fairness, such as this library, are simplistic and limited in actually achieving fairness in real-world decision processes.

More information on these limitations can be found [here](https://dl.acm.org/doi/full/10.1145/3624700) or [here](https://ojs.aaai.org/index.php/AAAI/article/view/26798).

## Future plans
The library maintains a core focus on only fairrets for now, yet we plan to add more fairness tools that align with the design principles in the future. These may involve breaking changes. At the same time, we'll keep reviewing the role of this library within the wider ecosystem of fairness toolkits. 

Want to help? Please don't hesitate to open an issue, draft a pull request, or shoot an email to [maarten.buyl@ugent.be](mailto:maarten.buyl@ugent.be).

## Citation
This framework will be presented as a paper at ICLR 2024. If you found this library useful in your work, please consider citing it as follows:

```bibtex
@inproceedings{buyl2024fairret,
    title={fairret: a Framework for Differentiable Fairness Regularization Terms},
    author={Buyl, Maarten and Defrance, Marybeth and De Bie, Tijl},
    booktitle={International Conference on Learning Representations},
    year={2024}
}
```
