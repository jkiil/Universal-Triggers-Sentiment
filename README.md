# Universal Adversarial Triggers for Attacking Sentiment Analysis

This repo is based on the official code for the EMNLP 2019 paper, Universal Adversarial Triggers for Attacking and Analyzing NLP, found [here](https://github.com/Eric-Wallace/universal-triggers). This repository contains the code for replicating our experiments and creating universal triggers.

## Dependencies

This code is written using PyTorch. The code for GPT-2 is based on [HuggingFace's Transformer repo](https://github.com/huggingface/pytorch-transformers) and the experiments on SQuAD, SNLI, and SST use [AllenNLP](https://github.com/allenai/allennlp/). The code is flexible and should be generally applicable to most models (especially if its in AllenNLP), i.e., you can easily extend this code to work for the model or task you want. 

The code is made to run on GPU, and a GPU is likely necessary due to the costs of running the larger models. I used one GTX 960 for all the experiments; the sst.py experiments run in a few minutes, and the other models can take up to half an hour.

## Installation

An easy way to install the code is to create a fresh anaconda environment:

```
conda create -n triggers python=3.6
```
The most difficult step is installing torch 1.1.0, which is required for some of our legacy packages.
Some suggested methods for Windows OS are described in requirements.txt.
```
pip install -r requirements.txt
```
## Getting Started

The `sst` folder contains our scripts to generate triggers that attack sentiment classification.
+ `sst.py` contains three parameters: model_type, dataset_label_filter, and test_triggers, which are described at the bottom of the file. It walks through training a simple GloVe-based LSTM/GRU sentiment analysis model that can be used to generate and test triggers.
+ `sst_amazon.py` contains two parameters: dataset_label_filter and test_triggers, and can be used to generate and test LSTM triggers based on HuggingFace's amazon_reviews_multi dataset.
+ `sst_finance.py` contains two parameters: dataset_label_filter and test_triggers, and can be used to generate and test LSTM triggers based on HuggingFace's financial_phrasebank dataset.
+ `sst_twitter.py` contains two parameters: dataset_label_filter and test_triggers, and can be used to generate and test LSTM triggers based on the Sentiment140 dataset.

The gradient-based attacks are written in `attacks.py`. 
The file `utils.py` contains the code for evaluating models, computing gradients, and evaluating the top candidates for the attack.
The `tmp` folder contains some of the vocab and models that we generated. Feel free to delete the files and re-run to get your own results.