from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from input_output import read_input
import pandas as pd

features_train, labels_train, features_test = read_input()

scores = []
clf1 = DecisionTreeClassifier()
clf2 = GaussianNB()
clf3 = SVC(kernel='rbf', C=5)#83.8
clf4 = RandomForestClassifier(n_estimators=20)

scores.append(cross_val_score(clf1, features_train, labels_train, cv=5).mean())
scores.append(cross_val_score(clf2, features_train, labels_train, cv=5).mean())
scores.append(cross_val_score(clf3, features_train, labels_train, cv=5).mean())
scores.append(cross_val_score(clf4, features_train, labels_train, cv=5).mean())
print scores

# final submission file
clf3.fit(features_train, labels_train)
pred = clf3.predict(features_test)
pd.concat([pd.DataFrame({'PassengerId': range(892, 1310, 1)}), pd.DataFrame({'Survived': pred.tolist()})], axis=1,
          join='inner').to_csv('final_submission.csv', index=False)
