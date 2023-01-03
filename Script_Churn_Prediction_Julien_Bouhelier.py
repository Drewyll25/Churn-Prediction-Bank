# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 17:07:31 2022

@author: julie
"""
# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %% Data overview
# In this section we will seek to explore the structure of our data

# Import Data
df = pd.read_csv(
    'C:/Users/julie/OneDrive/Documents/TBS/M2/Stage Quinten/UC_banque_candidat_DSJ/df.csv',
    sep=';')
print(df.info())
print(df.shape)

# %Check columns list and missing values
print(df.isnull().sum())
# The dataset which results of dropping all Nan values is only 2065 rows,
# Nan values are widespread.

# Get unique count for each variable
print(df.nunique())

# Review the top rows of what is left of the data frame
print(df.head())

# Check variable data types
df.dtypes
# So we 19 have categorical variables and 44 continuous variables

# Change espace client into objects (for data viz)
df['espace_client_web'].replace((1, 0), ('oui', 'non'), inplace=True)

# Change interet_compte_epargne_total into floats
df['interet_compte_epargne_total'].replace(
    ('', ' ', '  '), np.nan, inplace=True)
df['interet_compte_epargne_total'] = df['interet_compte_epargne_total'].astype(
    float)

# %% Exploratory Data Analysis

# Variables influence on churn
labels = 'Exited', 'Retained'
sizes = [df.churn[df['churn'] == "oui"].count(),
         df.churn[df['churn'] == "non"].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customer churned and retained", size=20)
plt.show()

# We first review the 'Status' relation with categorical variables
# Genre
sns.countplot(x='genre', hue='churn', data=df)

# Contact
fig1, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x='espace_client', hue='churn', data=df, ax=axarr[0][0])
sns.countplot(x='espace_client_web', hue='churn', data=df, ax=axarr[0][1])
sns.countplot(x='methode_contact', hue='churn', data=df, ax=axarr[1][0])
sns.countplot(x='branche', hue='churn', data=df, ax=axarr[1][1])

# General
fig2, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x='banque_principale', hue='churn', data=df, ax=axarr[0][0])
sns.countplot(x='cartes_bancaires', hue='churn', data=df, ax=axarr[0][1])
sns.countplot(x='type', hue='churn', data=df, ax=axarr[1][1])
sns.countplot(x='segment_client', hue='churn', data=df, ax=axarr[1][0])

# Comptes
fig3, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x='compte_courant', hue='churn', data=df, ax=axarr[0][0])
sns.countplot(x='compte_epargne', hue='churn', data=df, ax=axarr[0][1])
sns.countplot(x='compte_titres', hue='churn', data=df, ax=axarr[1][1])
sns.countplot(x='PEA', hue='churn', data=df, ax=axarr[1][0])

# Assurances
fig4, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x='assurance_habitation', hue='churn', data=df, ax=axarr[0][0])
sns.countplot(x='assurance_vie', hue='churn', data=df, ax=axarr[0][1])
sns.countplot(x='assurance_auto', hue='churn', data=df, ax=axarr[1][1])
sns.countplot(x='assurance_habitation', hue='churn', data=df, ax=axarr[1][0])

# Credits
fig5, axarr = plt.subplots(2, figsize=(12, 12))
sns.countplot(x='credit_immo', hue='churn', data=df, ax=axarr[0])
sns.countplot(x='credit_autres', hue='churn', data=df, ax=axarr[1])

# Relations based on the continuous data attributes
fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.boxplot(y='age', x='churn', hue='churn', data=df, ax=axarr[0][0])
sns.boxplot(
    y='anciennete_mois',
    x='churn',
    hue='churn',
    data=df,
    ax=axarr[0][1])
sns.boxplot(y='agios_6mois', x='churn', hue='churn', data=df, ax=axarr[1][0])
sns.boxplot(
    y='interet_compte_epargne_total',
    x='churn',
    hue='churn',
    data=df,
    ax=axarr[1][1])

# We are going to drop genre, branch and credits because they dont affect churn

# %% Data Prep for Machine Learning

## Arrange the dataset ##
# Arrange columns by data type for easier manipulation
continuous_vars = ['age', 'anciennete_mois', 'agios_6mois',
                   'interet_compte_epargne_total']


categorical_vars = ['banque_principale', 'espace_client_web', 'compte_epargne',
                    'compte_titres', 'assurance_vie', 'segment_client',
                    'methode_contact', 'cartes_bancaires', 'espace_client',
                    'compte_courant', 'PEA', 'assurance_habitation',
                    'assurance_auto', 'genre', 'branche', 'credit_autres',
                    'credit_immo', 'type']

anonimyzed_vars = [
    'var_0',
    'var_1',
    'var_2',
    'var_3',
    'var_4',
    'var_5',
    'var_6',
    'var_7',
    'var_8',
    'var_9',
    'var_10',
    'var_11',
    'var_12',
    'var_13',
    'var_14',
    'var_15',
    'var_16',
    'var_17',
    'var_18',
    'var_19',
    'var_20',
    'var_21',
    'var_22',
    'var_23',
    'var_24',
    'var_25',
    'var_26',
    'var_27',
    'var_28',
    'var_29',
    'var_30',
    'var_31',
    'var_32',
    'var_33',
    'var_34',
    'var_35',
    'var_36',
    'var_37',
    'var_38',
]

# Create new dataset
dataset = df
dataset = dataset[['churn'] +
                  continuous_vars +
                  categorical_vars +
                  ['id_client']]


# %%
# Drop the variables which dosent affect the churn
dataset = dataset.drop(
    ['genre', 'branche', 'credit_autres', 'credit_immo'], axis=1)

# Drop type because to many Na values
dataset = dataset.drop(['type'], axis=1)

# Drop the rest of NA values
dataset = dataset.dropna()

#%%Convert the categorical variables ##
# For booleans, we are going to replace oui by 1 and non by 0
dataset['churn'].replace(('oui', 'non'), (1, 0), inplace=True)
dataset['churn'] = pd.to_numeric(dataset['churn'])

dataset['espace_client_web'].replace(('oui', 'non'), (1, 0), inplace=True)
dataset['espace_client_web'] = pd.to_numeric(dataset['espace_client_web'])

dataset['assurance_vie'].replace(('oui', 'non'), (1, 0), inplace=True)
dataset['assurance_vie'] = pd.to_numeric(dataset['assurance_vie'])

dataset['banque_principale'].replace(('oui', 'non'), (1, 0), inplace=True)
dataset['banque_principale'] = pd.to_numeric(dataset['banque_principale'])

dataset['compte_epargne'].replace(('oui', 'non'), (1, 0), inplace=True)
dataset['compte_epargne'] = pd.to_numeric(dataset['compte_epargne'])

dataset['compte_titres'].replace(('oui', 'non'), (1, 0), inplace=True)
dataset['compte_titres'] = pd.to_numeric(dataset['compte_titres'])

# For variables with more than two categories, lets create dummis
methode_contact = pd.get_dummies(
    dataset.methode_contact, 'contact', '_').iloc[:, 1:]
segment_client = pd.get_dummies(
    dataset.segment_client, 'client', '_').iloc[:, 1:]
cartes_bancaires = pd.get_dummies(
    dataset['cartes_bancaires'],
    'cartes_bancaires',
    '_')
dataset = pd.concat(
    [dataset, methode_contact, segment_client, cartes_bancaires], axis=1)

# %% Drop the old non dummies variables
dataset = dataset.drop(dataset.iloc[:, 10:13], axis=1)

# %% Drop the variables where for every prenium credit card we have no information

# drop variables with two many inconnu variables
dataset = dataset.drop(['espace_client',
                        'compte_courant', 'PEA',
                        'assurance_habitation',
                        'assurance_auto'], axis=1)

# Check nan values
dataset.isnull().sum()

# Export the dataset dataset_prenium
dataset.to_csv(
    r'C:/Users/julie/OneDrive/Documents/TBS/M2/Stage Quinten/UC_banque_candidat_DSJ/dataset.csv',
    index=False)

# %% Data Processing
# Import Data
dataset = pd.read_csv(
    'C:/Users/julie/OneDrive/Documents/TBS/M2/Stage Quinten/UC_banque_candidat_DSJ/dataset.csv')


# Delete the ID por modeling
dataset_with_ID = dataset
dataset = dataset.drop(['id_client', ], axis=1)

# %% Watch Correlations with final variables
correlations = dataset.corrwith(dataset.churn)
correlations = correlations[correlations != 1]
correlations.plot.bar(
    figsize=(30, 10),
    fontsize=15,
    color='#ec838a',
    rot=90, grid=True)
plt.title('Correlation with Churn Rate \n',
          horizontalalignment="center", fontstyle="normal",
          fontsize="22", fontfamily="sans-serif")


# %% Set and compute the Correlation Matrix:
sns.set(style="white")
corr = dataset.corr()
# Generate a mask for the upper triangle:
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure and a diverging colormap:
f, ax = plt.subplots(figsize=(20, 20))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio:
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# %% Calculate the VIF (Variance Inflation Factor)
# Check
def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i)
                  for i in range(X.shape[1])]
    return vif


calc_vif(dataset)


# %% Drop variables which are correlated
dataset.drop('agios_6mois', inplace=True, axis=1)
dataset.drop('cartes_bancaires_medium', inplace=True, axis=1)
dataset.drop('interet_compte_epargne_total', inplace=True, axis=1)

#%% ## Random Forest ##


# split with X and y
X = dataset.drop(['churn'], axis=1)
Y = dataset['churn']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# Machine Learning Algorithm Training
classifier = RandomForestClassifier(n_estimators=200, random_state=0)
classifier.fit(X_train, Y_train)
predictions = classifier.predict(X_test)

# Machine Learning Algorithm Evaluation

# Score of accuracy
print(accuracy_score(Y_test, predictions))

# %% confusion matrix
cm = metrics.confusion_matrix(Y_test, predictions)
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax)
# labels, title and ticks
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Actually Churn', 'Actually Stay'])
ax.yaxis.set_ticklabels(['Predicted Churn', 'business Stay'])

# %% Classification report
clf_report = classification_report(Y_test,
                                   predictions,
                                   output_dict=True)

sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)


# %%Most important variables
feat_importances = pd.Series(classifier.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')


#%% ## Logistic Regression ##
x = dataset.drop(['churn'], axis=1)
y = dataset['churn']

# Split in 80% / 20%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=50)

# Fit the model
logmodel = LogisticRegression(random_state=None)
logmodel.fit(x_train, y_train)
result=logmodel.fit(x_train, y_train)

# Predict the value for new, unseen data
pred = logmodel.predict(x_test)

# Machine Learning Algorithm Evaluation

# Score of accuracy
print(logmodel.score(x_test, y_test))

# %% confusion matrix
cm = metrics.confusion_matrix(y_test, pred)
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax)
# labels, title and ticks
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Actually Churn', 'Actually Stay'])
ax.yaxis.set_ticklabels(['Predicted Churn', 'business Stay'])

# %% Classification report
clf_report = classification_report(y_test,
                                   pred,
                                   output_dict=True)

sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)


# %% Give back the ID
dataset['Probality_of_churn'] = logmodel.predict_proba(
    dataset[x_test.columns])[:, 1]

dataset['id_client'] = dataset_with_ID['id_client']

Probability_of_churn = dataset[['id_client'] + ['Probality_of_churn']]

Probability_of_churn.to_csv(
    r'C:/Users/julie/OneDrive/Documents/TBS/M2/Stage Quinten/UC_banque_candidat_DSJ/Probability_of_churn.csv',
    index=False)
