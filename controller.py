from datasets import *
import numpy as np
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.linear_model.sparse import SGDClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

train_header, train = readFile('data/train.csv')
test_header, test = readFile('data/test.csv')

vectorizer = CountVectorizer(min_n=1, max_n=5)

X_train = vectorizer.fit_transform(train[:,2])
y_train = train[:,0].astype(int)

X_test = vectorizer.transform(test[:,1])

#model = MultinomialNB()
#model = SGDClassifier()
model = SVC()
scores = cross_validation.cross_val_score(model, X_train, y_train, cv=5)
print 'Score: ' + str(sum(scores) / len(scores))

model.fit(X_train, y_train)
preds = model.predict(X_test)
np.savetxt('submission.csv', preds, fmt='%s')
