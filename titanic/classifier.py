import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./train.csv')

features_train = np.array(df[['Pclass','Sex']].apply(LabelEncoder().fit_transform))
labels_train = np.array(df['Survived'])

X_train, X_test, Y_train, Y_test = train_test_split(features_train, labels_train, test_size=.25, random_state=42)

#clf = DecisionTreeClassifier()#.780
clf = GaussianNB()#.784
#clf = SVC(kernel="rbf", C=100)#.735
#clf = RandomForestClassifier(random_state=1, n_estimators=10)#.775
clf.fit(X_train, Y_train)

print accuracy_score(clf.predict(X_test), Y_test)

#final submission file
df_test = pd.read_csv('./test.csv')
features_test = np.array(df_test[['Pclass','Sex']].apply(LabelEncoder().fit_transform))
pred = clf.predict(features_test)
pd.concat([pd.DataFrame({'PassengerId': range(892, 1310, 1)}), pd.DataFrame({'Survived': pred.tolist()})], axis=1, join='inner').to_csv('final_submission.csv', index=False)






