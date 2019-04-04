## Hierarchical Dynamic Model (HDM)

Hierarchical dynamic model (HDM) is a probabilistic dynamic model which explicitly models spatial and temporal variations in the dynamic data. The temporal variation is handled in two aspects. First, we incorporate a probabilistic duration mechanism to allow flexible speed at each phase of an activity. Second, the transitions among different phases of an activity are modeled by transition probabilities among different hidden states. The spatial variation is modeled by probability distribution on observations in each individual frame. To further improve the capability of handling intra-class variation, we extend the model following the Bayesian framework, by allowing the parameters to vary across data, yielding a hierarchical structure.

## How to Use

This repository provides an implementation of HDM for classification task. It includes the following key components:

1. Hyperparameter learning

2. Bayesian inference (only Gibbs sampling via Matlab)

3. Classification

4. Analysis of uncertainty

To perform 1-3, run 'script_data_classification_hdm.m' script in Matlab (R2016a and up), follow the prompt in command window to select available dataset. 

To perform 4, run 'script_classification_uncertainty.m' in Matlab after running 1-3.

## Dependencies

Bayes Net Toolbox (BNT)

## Related Publication

The code and data are used to produce some of the experiment results reported in the following paper.

Rui Zhao, Wanru Xu, Hui Su and Qiang Ji, "Bayesian Hierarchical Dynamic Model for Human Action Recognition," IEEE Conference on Computer Vision Pattern Recognition (CVPR), 2019.

## License Condition

Copyright (C) 2019 Rui Zhao 

Distribution code version 1.0 - 4/3/2019. This code is for research purpose only.
