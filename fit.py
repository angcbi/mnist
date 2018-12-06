import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


train_X = np.load('./data/train.npy')
train_y = np.load('./data/train_label.npy')
test_X = np.load('./data/test.npy')
test_y = np.load('./data/test_label.npy')


lr = LogisticRegression(solver='lbfgs', C=100, multi_class='multinomial')
lr.fit(train_X, train_y)
joblib.dump(lr, './model/lr.model')

y_hat_lr = lr.predict(test_X)
print('---lr----')
print('acc', accuracy_score(test_y, y_hat_lr))
# 混淆矩阵，行是真实样本，列是预测值
print('cfm', confusion_matrix(test_y, y_hat_lr))

rf = RandomForestClassifier(n_estimators=400, max_depth=40, oob_score=True, n_jobs=10)
rf.fit(train_X, train_y)
joblib.dump(rf, './model/rf.model')

y_hat_rf = rf.predict(test_X)
print('---rf----')
print('oob score', rf.oob_score_)
print('acc', accuracy_score(test_y, y_hat_rf))
# 混淆矩阵，行是真实样本，列是预测值
print('cfm', confusion_matrix(test_y, y_hat_rf))

ada_rf = AdaBoostClassifier(n_estimators=500, learning_rate=1)
ada_rf.fit(train_X, train_y)
joblib.dump(ada_rf, './model/ada_rf.model')

y_hat_ada_rf = ada_rf.predict(test_X)
print('---y_hat_ada_rf----')
print('acc', accuracy_score(test_y, y_hat_ada_rf))
# 混淆矩阵，行是真实样本，列是预测值
print('cfm', confusion_matrix(test_y, y_hat_ada_rf))
