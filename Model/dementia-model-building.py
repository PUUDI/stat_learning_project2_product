import pandas as pd

dataset = pd.read_csv('https://gist.githubusercontent.com/PUUDI/861771ffca8462507b487b6f75f2386d/raw/44e4760f1f6ee628c9674fe1c87e63bd4fbcf19d/gistfile1.txt')
classes = ['Nondemented', 'Demented']

#dropping the converted Group data points
dataset.drop(dataset.loc[dataset['Group'] == 'Converted'].index, inplace=True) # dataset.Group[dataset.Group == 'Converted'] = 'Nondemented' 

#Dropping the unwanted columns that won't be needing to include in our model
dataset.drop(['Subject ID', 'MRI ID', 'Hand','CDR','MR Delay','Visit'], axis=1, inplace=True) # 'MR Delay''Visit','Age','EDUC','eTIV'

#Encoding binary variables
dataset['M/F'] = dataset['M/F'].apply(lambda x: ['M', 'F'].index(x))

#Encoding the class variable
dataset['Class'] = [classes.index(group) for group in dataset['Group']]
# Feature Encoding

#Missing value Imputation
from feature_engine.imputation import RandomSampleImputer
#Imputing mean values for missing values
# set up the imputer
imputer = RandomSampleImputer(
        random_state=['SES','MMSE'],
        seed='observation',
        seeding_method='add'
    )

# fit the imputer
imputer.fit(dataset)

dataset = imputer.transform(dataset)

y = dataset.Class
dataset.drop(['Group','Class'], axis=1, inplace=True) # 'MR Delay''Visit',
X = dataset

#Oversampling
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from numpy import where

counter = Counter(y)
print(counter)
# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter)

from sklearn.model_selection import train_test_split
train_feature, test_feature, train_label, test_label = train_test_split(X, y, test_size=0.3, random_state=42)

#Model fitting
from sklearn.ensemble import RandomForestClassifier
rf_final = RandomForestClassifier(bootstrap=False,max_depth=None,max_features='sqrt',min_samples_leaf=1,min_samples_split=2,n_estimators=750,random_state=40)
rf_final.fit(train_feature, train_label)


import pickle
pickle.dump(rf_final, open('penguins_clf.pkl', 'wb'))





# Ordinal feature encoding
# # https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
# df = penguins.copy()
# target = 'species'
# encode = ['sex','island']

# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix=col)
#     df = pd.concat([df,dummy], axis=1)
#     del df[col]

# target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
# def target_encode(val):
#     return target_mapper[val]

# df['species'] = df['species'].apply(target_encode)

# # Separating X and y
# X = df.drop('species', axis=1)
# Y = df['species']

# # Build random forest model
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier()
# clf.fit(X, Y)

# Saving the model
# import pickle
# pickle.dump(clf, open('penguins_clf.pkl', 'wb'))
