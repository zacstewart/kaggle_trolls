from datasets import *
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV

train_header, train = readFile('data/train.csv')
test_header, test = readFile('data/test.csv')

#model = MultinomialNB()
#model = SGDClassifier()

model_pipeline = Pipeline([
  ('vect', CountVectorizer()),
  ('tfidf', TfidfTransformer()),
  ('lsvc', LinearSVC()),
])

parameters = {
  'vect__max_n': (1, 2),
  'tfidf__use_idf': (True, False),
  'lsvc__C': (100, 1000),
}

model_gs = GridSearchCV(model_pipeline, parameters, n_jobs=-1)

text_train = train[:,2]
y_train    = train[:,0].astype(int)
text_test  = test[:,1]

scores = cross_validation.cross_val_score(model_gs, text_train, y_train, cv=5)
print 'Score: ' + str(sum(scores) / len(scores))

model_gs.fit(text_train, y_train)
preds = model_gs.predict(text_test)
np.savetxt('submission.csv', preds, fmt='%s')
