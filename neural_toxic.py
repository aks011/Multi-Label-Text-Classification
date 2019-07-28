# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 16:03:26 2019

@author: ARPIT
"""

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.neural_network import MLPClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC
#from sklearn.naive_bayes import MultinomialNB

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')
#train_target = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

train_text = train['comment_text'].str[:400]
test_text = test['comment_text']
#all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=4000)
word_vectorizer.fit(train_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=20000)
char_vectorizer.fit(train_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
#submission = pd.DataFrame.from_dict({'id': test['id']})
#train_target = train[class_name]
#classifier =  MLPClassifier(hidden_layer_sizes=(8,),batch_size=64)
#classifier.fit(train_features, train_target)
for class_name in class_names:
    train_target = train[class_name]
    classifier =  MLPClassifier(hidden_layer_sizes=(16,),batch_size=64,max_iter=400)
    classifier.fit(train_features, train_target)

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    #classifier.fit(train_features, train_target)
#    submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

#submission.to_csv('submission.csv', index=False) 


#for class_name in class_names:
#    train_target = train[class_name]
#    classifier =  MultinomialNB().fit(train_features, train_target) 
#    print(classifier.predict(test_features))