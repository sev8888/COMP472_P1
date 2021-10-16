import numpy                    as np
import pandas                   as pd
import matplotlib.pyplot        as plt

from pandas.api.types           import is_numeric_dtype
from sklearn.model_selection    import train_test_split

from sklearn.naive_bayes        import GaussianNB
from sklearn.tree               import DecisionTreeClassifier
from sklearn.model_selection    import GridSearchCV
from sklearn.linear_model       import Perceptron
from sklearn.neural_network     import MLPClassifier

# -----------
# PART 2 - load dataset in python
# -----------
drug200data = pd.read_csv("drug200.csv")    # import dataset "drug200.csv" as "drug200data"
features = drug200data.columns.values       # get features of dataset

# -----------
# PART 3 - plot distribution
# -----------
drugList = drug200data[features[-1]].values # storing all the Drug value from Drug colume to the DrugList
drugType = []                               # save drugTypes and drugCounts in dynamic list
drugCount = []

# Accessing each type of drug in DrugList and count total amount of each drug
for drug in drugList:
    for i in range(len(drugType)):
        if drug == drugType[i]:
            drugCount[i] = drugCount[i] + 1
            break
    else:
        drugType.append(drug)
        drugCount.append(1)

# sort drugs in alphabetical order
drugType, drugCount = zip(*sorted(zip(drugType,drugCount)))

# plot drug distribution on the bar chart
#print(drugCount2)
plt.bar(drugType,drugCount)
plt.title("Distribution of Drugs by Type")
plt.savefig('drug-distribution.pdf')

# -----------
# PART 4 - convert ordinal and nominal features in numerical format
# -----------
for f in features:
    if not is_numeric_dtype(drug200data[f]):
        fvalues = drug200data[f].unique()
        #if ordinal feature, convert using [LOW, NORMAL, HIGH] scale
        if 'HIGH' in fvalues or 'NORMAL' in fvalues or 'LOW' in fvalues:
            drug200data[f] = pd.Categorical(values=drug200data[f], categories=['LOW','NORMAL','HIGH'], ordered=True)
        #else convert as nominal feature
        else:
            drug200data[f] = pd.Categorical(values=drug200data[f], ordered=False)
        print("{}: {}".format(f, drug200data[f].cat.categories))
        drug200data[f] = drug200data[f].cat.codes

# -----------
# PART 5 - split dataset
# -----------
x = drug200data[features[0:-1]]
y = drug200data[features[-1]]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=10)

# -----------
# PART 6 - run classifiers
# -----------

# Naive Bayes Classifier
modelNB = GaussianNB()
modelNB.fit(x_train,y_train)
modelNB.score(x_test,y_test)
naiveBayesPredict = modelNB.predict(x_test)
naiveBayesPredictProbability = modelNB.predict_proba(x_test)

# Base-DT
modelDT = DecisionTreeClassifier()
modelDT.fit(x_train,y_train)
modelDT.score(x_test,y_test)
BaseDTPredict = modelDT.predict(x_test)
BaseDTPredictProbability = modelDT.predict_proba(x_test)
