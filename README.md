# NLP modeling case

## Description

The task is to use a version of the Twitter sentiment dataset (You can download dataset through the link: [The dataset](https://drive.google.com/file/d/13mAaFqCrscUYkoITf4rZ6qG9ptAlIJVb/view?usp=sharing)) and create a simple but complete preprocessing, training and validation pipeline to predict the target *is_positive*.


## Solution
The repository contains a model for sentiment analysis using the Support Vector Machine (SVM) classifier. The goal is to predict the sentiment labels of text messages as either positive or negative with a supervised method. The smaller version of dataset was used for training this model.

## Requirements
To run the code you need to install the libraries below:

```bash
pip install pandas
pip install scikit-learn
pip install nltk
```

## Preprocessing
Before starting the training process the dataset needs to undergo preprocessing. The following preprocessing steps are applied:
- Lowercasing
- Tokenization and punctuation removal
- Stopword removal
- Lemmatization

## Training and Evaluation
For the purpose of feature extraction, the text data is transformed into numerical features using the TF-IDF vectorization technique provided by the `TfidfVectorizer` class from scikit-learn. (I also recommend Word2vec word embeddings solution for short texts istead of TF-IDF vectorization technique.)

The SVM classifier is trained on the training set. The trained model is then used to predict the sentiment labels for the test set.

The evaluation metrics calculated are as follows:
- Accuracy
- Precision
- Recall
- F1 Score

The accuracy of this model is 73%.