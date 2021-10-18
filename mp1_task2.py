import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# -----------
# PART 2 - load dataset in python
# -----------
drug200data = pd.read_csv("drug200.csv")    # import dataset "drug200.csv" as "drug200data"
features = drug200data.columns.values       # get features of dataset

# -----------
# PART 3 - plot distribution
# -----------
drugList = drug200data[features[-1]] # storing all the Drug value from Drug colume to the DrugList
drugCount = drugList.value_counts().sort_index().reset_index()
drugTypes = drugCount[drugCount.axes[1][0]].values
print(drugTypes)
drugCount = drugCount[drugCount.axes[1][1]].values
print(drugCount)

# plot drug distribution on the bar chart
plt.bar(drugTypes,drugCount)
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
        drug200data[f] = drug200data[f].cat.codes

# -----------
# PART 5 - split dataset
# -----------
X = drug200data[features[0:-1]]
y = drug200data[features[-1]]
x_train, x_test, y_train, y_test = train_test_split( X, y )
answer = y_test

# -----------
# PART 6 - run classifiers on dataset
# -----------

def runClassifiers(version):
    # Naive Bayes Classifier
    NB = GaussianNB()
    NB.fit(x_train,y_train)
    NBpredict = NB.predict(x_test)
    save_results("NB classifier, try "+str(version), NBpredict, answer)

    # Base Decision Tree
    DT = DecisionTreeClassifier()
    DT.fit(x_train,y_train)
    BaseDTpredict = DT.predict(x_test)
    save_results("Decision Tree, try "+str(version), BaseDTpredict, answer)

    # Top Decision Tree
    DTparams = {'criterion':( 'gini', 'entropy' ),
                'max_depth':( 10, 20 ),                     #add 2 diff values here (default=None)
                'min_samples_split':( 0.5, 1, 5 )}         #add 3 diff values here (default=2)
    TOPDT = GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=DTparams)
    TOPDT.fit(x_train,y_train)
    TOPDTpredict = TOPDT.predict(x_test)
    save_results("Top Decision Tree, try "+str(version), TOPDTpredict, answer, params=TOPDT.best_params_)

    # PER
    PER = Perceptron()
    PER.fit(x_train,y_train)
    PERpredict = PER.predict(x_test)
    save_results("Perceptron, try "+str(version), PERpredict, answer)

    # Base MLP
    MLP = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='sgd')
    MLP.fit(x_train,y_train)
    MLPpredict = MLP.predict(x_test)
    save_results("Multilayer Perceptron, try "+str(version), MLPpredict, answer, params=MLP.get_params())

    # Top MLP
    MLPparams = {'activation':( 'logistic', 'tanh', 'relu', 'identity' ),
                 'hidden_layer_sizes':( (50,50), (20,20,20,20,20) ), #add 2 diff network architecture
                 'solver':( 'adam', 'sgd' )}
    TOPMLP = GridSearchCV(estimator=MLPClassifier(),param_grid=MLPparams)
    TOPMLP.fit(x_train,y_train)
    TOPMLPpredict = TOPMLP.predict(x_test)
    save_results("Top Multilayer Perceptron, try "+str(version), MLPpredict, answer, params=TOPMLP.best_params_)

# -----------
# PART 7 -  save results to drug-performance.txt
# -----------

def save_results(name, pred, answ, params=None):
    with open('drug-performance.txt','a') as f:
        #===================================================================================
        f.writelines( "(a) :: === {} ===\n".format(name))
        if (params != None):
            f.writelines("{}\n".format(params))
        #===================================================================================
        cfs_matrix = confusion_matrix(y_pred = pred, y_true = answ)
        f.writelines( "\n(b) :: confusion matrix\n\n"
                    + np.array2string(cfs_matrix) + "\n")
        #===================================================================================
        report = classification_report(y_pred = pred, y_true = answ, target_names = drugTypes)
        f.writelines( "\n(c) :: per class precision, recall and F1-measure\n\n{}".format(report))
        #===================================================================================
        f.writelines( "\n(d) :: accuracy, macro-average F1 and weighted-average F1\n\n"
                    + "{:>15} : {:.6f}\n".format("Accuracy", accuracy_score(y_pred= pred, y_true = answ))
                    + "{:>15} : {:.6f}\n".format("Macro-F1", f1_score(y_pred= pred, y_true = answ, average = 'macro'))
                    + "{:>15} : {:.6f}\n".format("Weighted-F1", f1_score(y_pred= pred, y_true = answ, average = 'weighted')))
        #===================================================================================
        f.writelines("\n")
    f.close()

if os.path.exists("drug-performance.txt"):
    os.remove("drug-performance.txt")

for i in range(10):
    print("try {}".format(i+1))
    runClassifiers(i+1)

print   ("===============\n"
        +"End of program.\n"
        +"===============")