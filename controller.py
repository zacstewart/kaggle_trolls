from datasets import *
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV

from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid

train_header, train = readFile('data/train.csv')
test_header, test   = readFile('data/test.csv')

text_train, y_train = train[:,2], train[:,0].astype(int)
text_test  = test[:,1]

models = {
    'lsvc': {'algo': LinearSVC,             'params': [('C',  (100,  1000))]},
    'svc':  {'algo': SVC,                   'params': [('C',  (100,  1000))]},
    'sgdc': {'algo': SGDClassifier,         'params': []},
    'perc': {'algo': Perceptron,            'params': []},
    'bnb':  {'algo': BernoulliNB,           'params': []},
    'mnb':  {'algo': MultinomialNB,         'params': []},
    'knc':  {'algo': KNeighborsClassifier,  'params': []},
    'nc':   {'algo': NearestCentroid,       'params': []}
}

def makePipeline(name, model, gs=True):
  print 'Training ' + name + '...'
  pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    (name, model['algo']()),
  ])

  parameters = {'vect__max_n': (1, 2), 'tfidf__use_idf': (True, False)}
  model_params = dict([(name + '__' + pname, pvals) for pname, pvals in model['params']])
  parameters = dict(parameters.items() + model_params.items())

  if gs:
    return GridSearchCV(pipeline, parameters, n_jobs=-1)
  else:
    return pipeline

train_preds = np.zeros((len(train), len(models) + 1), dtype= int)
test_preds  = np.zeros((len(test), len(models)), dtype= int)
i = 0
for name, model in models.items():
  instance = makePipeline(name, model, gs=True)

  kf = KFold(len(train), k=5)
  for construct_indices, cv_indices in kf:
    construct, cv = train[construct_indices], train[cv_indices]

    text_construct, y_construct = construct[:,2], construct[:,0].astype(int)
    text_cv, y_cv               = cv[:,2], cv[:,0].astype(int)

    instance.fit(text_construct, y_construct)
    preds = instance.predict(text_cv)
    train_preds[cv_indices, 0] = y_cv
    train_preds[cv_indices, i+1] = preds

  score = cross_val_score(instance, text_train, y_train, cv=5)
  avg_score = sum(score) / len(score)
  print avg_score

  instance.fit(text_train, y_train)
  preds = instance.predict(text_test)
  test_preds[:,i] = preds

submission = None
best_score = 0
for name, model in models.items():
  print 'Super learning with ' + name + '...'
  if len(model['params']) > 0:
    instance = GridSearchCV(model['algo'](), dict(model['params']), n_jobs=1)
  else:
    instance = model['algo']()
  score = cross_val_score(instance, train_preds[:,1:], train_preds[:,0], cv=5)
  avg_score = sum(score) / len(score)
  print avg_score
  if avg_score > best_score:
    best_score = avg_score
    instance.fit(train_preds[:,1:], train_preds[:,0])
    submission = instance.predict(test_preds).astype(int)

np.savetxt('submission.csv', submission, fmt='%s')
