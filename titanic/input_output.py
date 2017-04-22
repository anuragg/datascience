import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


def read_input():
    df_train = pd.read_csv('./train.csv')
    df_test = pd.read_csv('./test.csv')

    #print df_train.describe()

    #print df_train.Sex.value_counts()
    #sns.factorplot('Sex', 'Survived', data=df_train)
    #plt.show()

    #print df_train.Pclass.value_counts()
    #sns.factorplot('Pclass', 'Survived', data=df_train)
    #plt.show()

    label_encoder = LabelEncoder()
    df_train['Sex'] = label_encoder.fit_transform(df_train['Sex'])
    df_test['Sex'] = label_encoder.fit_transform(df_test['Sex'])

    df_train['Age'] = df_train.apply(lambda x: 5 if np.isnan(x['Age']) and "aster" in x['Name'] else 29 if np.isnan(x['Age']) and "aster" not in x['Name'] else x['Age'], axis=1)
    df_test['Age'] = df_test.apply(
        lambda x: 5 if np.isnan(x['Age']) and "aster" in x['Name'] else 29 if np.isnan(x['Age']) and "aster" not in x[
            'Name'] else x['Age'], axis=1)

    #s = sns.FacetGrid(df_train, hue='Survived', aspect=3)
    #s.map(sns.kdeplot, 'Age', shade=True)
    #s.set(xlim=(0, 20))
    #s.add_legend()
    #plt.show()

    #label_encoder2 = LabelEncoder()
    #df_train['Cabin'] = label_encoder2.fit_transform(df_train['Cabin'])
    #df_test['Cabin'] = label_encoder2.fit_transform(df_test['Cabin'])

    label_encoder1 = LabelEncoder()
    df_train['Embarked'] = label_encoder1.fit_transform(df_train['Embarked'])
    df_test['Embarked'] = label_encoder1.fit_transform(df_test['Embarked'])

    #print df_train.Embarked.value_counts()
    #sns.factorplot('Embarked', 'Survived', data=df_train)
    #plt.show()

    df_test['Fare'] = df_test['Fare'].fillna(df_train['Fare'].mean())

    df_train['Mix'] = df_train['SibSp'] + df_train['Parch']
    df_test['Mix'] = df_test['SibSp'] + df_test['Parch']

    #print df_train.Mix.value_counts()
    #s = sns.FacetGrid(df_train, hue='Survived', aspect=3)
    #s.map(sns.kdeplot, 'Mix', shade=True)
    #s.set(xlim=(0, 5))
    #s.add_legend()
    #plt.show()

    df_train['is_1_2_3'] = df_train['Mix'].apply(is_1_2_3)
    df_test['is_1_2_3'] = df_test['Mix'].apply(is_1_2_3)

    df_train['fare_under_30'] = df_train['Fare'].apply(fare_under_30)
    df_test['fare_under_30'] = df_test['Fare'].apply(fare_under_30)

    df_train['age_under_15'] = df_train['Age'].apply(age_under_15)
    df_test['age_under_15'] = df_test['Age'].apply(age_under_15)

    features_train = np.array(df_train[['Pclass', 'Sex', 'age_under_15',  'fare_under_30', 'is_1_2_3', 'Embarked']])
    features_test = np.array(df_test[['Pclass', 'Sex', 'age_under_15', 'fare_under_30', 'is_1_2_3', 'Embarked']])
    labels_train = np.array(df_train['Survived'])

    return features_train, labels_train, features_test


def fare_under_30(row):
    result = 0.0
    if row<30:
        result = 1.0
    return result


def age_under_15(row):
    result = 0.0
    if row<15:
        result = 1.0
    return result


def is_1_2_3(row):
    result = 0.0
    if row==1 or row==2 or row==3:
        result = 1.0
    return result


#read_input()