# Quora Insincere Questions Classification

Code for the Kaggle competition [Quora Insincere Questions Classification](https://www.kaggle.com/c/quora-insincere-questions-classification)


## Overview

The goal of this competition is to build a binary classifier to predict whether or not a questions asked on Quora is insincere or not.  

Common reasons for a question to be considered insincere is if it has racist, sexist or other inappriopriate content, or it is phrased to convey inappriorate content instead of looking for answers.  

The training dataset has around 80K positive (i.e insincere) examples and 1.2M negative ones. The public test dataset has around 56K examples.

The model must be trained and the inference must be done on the test set within 2 hours on a machine run on the Kaggle platform which has 1 GPU.


## Experiments

### Baseline

The baseline is a multi-layer perceptron using as input the average of the Glove pre-trained word embeddings.

### Final model

The final model is a bidirectional LSTM with a multi-layer perceptron on top of it.
- The input is the concatenation of Glove, Paragram and Wikipedia pretrained-embeddings.
- The embedding layer is trainable.
- The output of the LSTM is reduced to a single vector by concatenation of average pooling and max pooling of the last hidden layer of the LSTM. The concatenation is passed as input to a multi-layer perceptron.
- A dropout is applied on the LSTM hidden layer(s).
- Adam optimizer is used with cross entropy loss.
- Using the entire training dataset with a maximum number of epochs found by doing 80/20 train/dev split and using early stopping to avoid overfitting.

### Other attempts

Other attempts were tried but not used in the end either because they did not help, there was not enough time for tuning them or because using them exceeds the time limit to train the model:
- Ensemble using the average prediction of 5 models from a 5-fold cross validation.
- Weight the loss to reduce the impact of imbalance either manually or using the inverse of the frequencies of each class.
- Threshold tuning to predict the final class by maximizing the score of the dev set.
- Using a "deep & wide" approach by concatenating general statistical features to the final embedding used for prediction. Features include number of out-of-vocabulary words, presence of common swearing, using words all in capital, etc.
- Apply lowercasing and/or lemmatization.


## Reproduce the results

### Setup Kaggle Docker image

See the Kaggle documentation for details.

### Download training set, test set and embeddings

The info below is also available on the Kaggle competition page.  
Put the training and test sets in the `../input/` directory.  
Put the embeddings in the `../input/embeddings/` directory.  

### Run script

``` bash
python3 quora_insincere_1.py
```
