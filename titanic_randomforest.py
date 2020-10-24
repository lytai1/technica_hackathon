import pandas as pd
from sklearn.model_selection._split import train_test_split
from sklearn import metrics
from sklearn.ensemble._forest import RandomForestClassifier
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import plot_roc_curve, roc_auc_score

def main():
    df = pd.read_csv("data/titanic.csv")
    print(df)
    df.drop(['Name'], axis=1, inplace=True)
    df['Sex'] = df['Sex'].map({'male':0, 'female': 1})
    print(df)

    # print("probabilities of survival:")
    # for col in df.columns[1:]:
    #     df2 = pd.crosstab(values=df.index, index=df['Survived'], columns=df[col], aggfunc='count')
    #     print(df2)


    features = list(df.columns[1:])
    labels = ['Survived', 'Not Survived']
    data = df[df.columns[1:]].values.tolist()
    target = list(df['Survived'].map({True:1, False:0}))
    print(len(features), "Features: ", features)
    print(len(data), 'Data: ', data)
    print(len(target), 'Target: ', target)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

    feature_imp = pd.Series(clf.feature_importances_,index=features).sort_values(ascending=True)
    #print(feature_imp)

    # Creating a bar plot
    sns.barplot(x=feature_imp, y=feature_imp.index)

    # Add labels to graph
    plt.xlabel('Feature Importance Score by Random Forest')
    plt.ylabel('Features')
    plt.legend()
    plt.show()

    # Predict probabilities for the test data.
    probs = clf.predict_proba(data)
    # Keep Probabilities of the positive class only.
    probs = probs[:, 1]
    # Compute the AUC Score.
    auc = roc_auc_score(target, probs)
    print('AUC: %.2f' % auc)

    # fpr, tpr, thresholds = roc_curve(y_test, probs)
    plot_roc_curve(clf, data, target)
    plt.show()

main()