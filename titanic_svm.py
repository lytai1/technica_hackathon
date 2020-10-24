import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn import svm
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np


def f_importances(coef, features, top=-1):
    imp = coef
    imp, features = zip(*sorted(zip(imp, features)))
    dic = dict(zip(imp, features))
    print(np.array(dic))
    print(imp)
    print(features)
    if top == -1:
        top = len(features)

    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), features[::-1][0:top])
    plt.show()


df = pd.read_csv("data/titanic.csv")
print(df)
df.drop(['Name'], axis=1, inplace=True)
df['Sex'] = df['Sex'].map({'male':0, 'female': 1})
print(df)

features = list(df.columns[1:])
labels = ['Survived', 'Not Survived']
data = df[df.columns[1:]].values.tolist()
target = list(df['Survived'].map({True:1, False:0}))
print(len(features), "Features: ", features)
print(len(data), 'Data: ', data)
print(len(target), 'Target: ', target)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2)

'''
# use this to find the optimal parameters for SVM
c = [0.01, 0.1, 1]
gamma = [0.01, 0.1, 1]
shrinking = [True, False]
degree = [3, 4, 5, 6, 7]

svm_grid = {'C': c, 'gamma' : gamma, 'shrinking' : shrinking, 'degree': degree}

svm1 = svm.SVC(kernel='linear')
svm_search = RandomizedSearchCV(svm1, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=30, verbose=1)
svm_search.fit(X_train, y_train)

print(svm_search.best_params_)
# {'shrinking': True, 'gamma': 1, 'degree': 3, 'C': 1}
'''

clf = svm.SVC(shrinking=True, kernel='linear', gamma=0.01, degree=4, C=1, probability=True) #svc setting for data with daily cases
# clf = svm.SVC(shrinking=True, kernel='linear', gamma=1, degree=6, C=0.1, probability=True) #svc setting for data with daily cases
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

#f_importances(clf.coef_, features)
f_importances(abs(clf.coef_[0]), features)


# # predict probabilities
svc_probs = clf.predict_proba(data)
# svc_probs = clf.predict_proba(X_test)

# keep probabilities for the positive outcome only
svc_probs = svc_probs[:, 1]
# calculate scores
svc_auc = roc_auc_score(target, svc_probs)
# svc_auc = roc_auc_score(X_test, svc_probs)
# summarize scores
print('svc: ROC AUC=%.3f' % (svc_auc))
# calculate roc curves
svc_fpr, svc_tpr, _ = roc_curve(target, svc_probs)
# svc_fpr, svc_tpr, _ = roc_curve(X_test, svc_probs)

# plot the roc curve for the model
plt.plot(svc_fpr, svc_tpr, marker='.', label='svc')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

#cross fold validation
scores = cross_val_score(clf, data, target, cv=5)
print(scores)