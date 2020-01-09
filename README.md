# Seq2VAR: multivariate time series representation with relational neural networks and linear autoregressive model
This repository presents the code for [our paper](https://hal.telecom-paristech.fr/hal-02293239) presented at the [AALTD workshop](https://project.inria.fr/aaltd19/accepted-papers/) of ECML/PKDD 2019 and appearing in the [Springer Lecture Notes in Artificial Intelligence proceedings 11986](https://www.springer.com/gp/book/9783030390976).

Notebooks "Seq2VAR_permutation.ipynb" and "Seq2VAR_homogeneous_springs.ipynb" contain the code for paper reproduction. Notebook "RNN_overfitting.ipynb" contains results for part 4.1

All the code to reproduce NRI (Kipf, 2018) is directly taken from the code of the original paper. 

### Abstract

Finding understandable and meaningful feature representation of multivariate time series (MTS) is a difficult task, since information is entangled both in temporal and spatial dimensions. In particular, MTS can be seen as the observation of simultaneous causal interactions between dynamical variables. Standard way to model these interactions is the vector linear autoregression (VAR). The parameters of VAR models can be used as MTS feature representation. Yet, VAR cannot generalize on new samples, hence  independent VAR models must be trained to represent different MTS. In this paper, we propose to use the inference capacity of neural networks to overpass this limit. We propose to associate a relational neural network to a VAR generative model to form an encoder-decoder of MTS. The model is denoted Seq2VAR for Sequence-to-VAR. We use recent advances in relational neural network to build our MTS encoder by explicitly modeling interactions between variables of MTS samples. We also propose to leverage reparametrization tricks for binomial sampling in neural networks in order to build a sparse version of Seq2VAR and find back the notion of Granger causality defined in sparse VAR models. We illustrate the interest of our approach through experiments on synthetic datasets.

