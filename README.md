# Universal Adversarial Triggers for Attacking Sentiment Classification

This repo is based on the official code for the EMNLP 2019 paper, Universal Adversarial Triggers for Attacking and Analyzing NLP, found [here](https://github.com/Eric-Wallace/universal-triggers). This repository contains the code for replicating our experiments and creating universal triggers.

## Dependencies

This code is written using PyTorch and [AllenNLP](https://github.com/allenai/allennlp/). The code is flexible and should be generally applicable to most models (especially if it's in AllenNLP), i.e., you can easily extend this code to work for the model or task you want. 

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

The `sst` folder contains our scripts to generate triggers that attack sentiment classification. Each of the files' main functions contain two parameters: dataset_label_filter and test_triggers, which are described at the bottom of each file. To run these scripts, first `cd` into the `sst` folder.
+ `sst_lstm.py` walks through training a simple GloVe-based LSTM sentiment analysis model that can be used to generate and test triggers.
+ `sst_gru.py` walks through training a simple GloVe-based GRU sentiment analysis model that can be used to generate and test triggers.
+ `sst_ffnn.py` walks through training a simple feedforward neural network bag-of-words sentiment analysis model that can be used to generate and test triggers.
+ `sst_amazon.py` can be used to generate and test LSTM triggers based on HuggingFace's amazon_reviews_multi dataset.
+ `sst_finance.py` can be used to generate and test LSTM triggers based on HuggingFace's financial_phrasebank dataset.
+ `sst_twitter.py` can be used to generate and test LSTM triggers based on the Sentiment140 dataset.

The root folder contains our Jupyter notebooks to fine-tune BERT models and test triggers.
+ `BERT_SST.ipynb` works with the Stanford Sentiment Treebank dataset.
+ `BERT_Finance.ipynb` works with HuggingFace's financial_phrasebank dataset.
+ `BERT_Amazon.ipynb` works with HuggingFace's amazon_reviews_multi dataset.
+ `BERT_Twitter.ipynb` works with the Sentiment140 dataset.

The gradient-based attacks are written in `attacks.py`, and `utils.py` contains the code for evaluating models, computing gradients, and evaluating the top candidates for the attack. These files are unchanged from the original [repo](https://github.com/Eric-Wallace/universal-triggers), with the exception of code added to filter the sentiment words listed in `positive_words.txt` and `negative_words.txt`.

The `tmp` folder contains some of the vocab and models that we generated. Feel free to delete the files and re-run to get your own results.

## References
The files `sst_lstm.py`, `attacks.py`, and `utils.py` were obtained from Eric Wallace's [reference repository](https://github.com/Eric-Wallace/universal-triggers), and are largely the same except for some minor modifications, such as the addition of sentiment filtering. We developed the rest of our code based on these three files, and while a lot of code is shared, they were written by us to suit our own unique experiments. All code related to the cleaning and loading of the domain-specific data sets was written by us. BERT model fine-tuning and testing can be found in `BERT_SST.ipynb,' 'BERT_Finance.ipynb,' 'BERT_Amazon.ipynb,' and 'BERT_Twitter.ipynb.'
