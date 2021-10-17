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
        print("{}: {}".format(f, drug200data[f].cat.categories))
        drug200data[f] = drug200data[f].cat.codes

# -----------
# PART 5 - split dataset
# -----------
x = drug200data[features[0:-1]]
y = drug200data[features[-1]]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20)

# -----------
# PART 6 - run classifiers
# -----------

def runClassifiers(x_train, x_test, y_train, y_test):
    # Naive Bayes Classifier
    NB = GaussianNB()
    NB.fit(x_train,y_train)
    NBpredict = NB.predict(x_test)

    # Base Decision Tree
    DT = DecisionTreeClassifier()
    DT.fit(x_train,y_train)
    BaseDTpredict = DT.predict(x_test)

    # Top Decision Tree
    DTparams = {'criterion':('gini','entropy'),
                'max_depth':(),                 #add 2 diff values here (default=None)
                'min samples split':()}         #add 3 diff values here (default=2)
    TOPDT = GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=DTparams)
    TOPDT.fit(x_train,y_train)
    TOPDTpredict = TOPDT.predict(x_test)

    # PER
    PER = Perceptron()
    PER.fit(x_train,y_train)
    PERpredict = PER.predict(x_test)

    # Base MLP
    MLP = MLPClassifier()
    MLP.fit(x_train,y_train)
    MLPpredict = MLP.predict(x_test)

    # Top MLP
    MLPparams = {'activation function':('sigmoid','tanh','relu','identity'),
                 'hidden_layer_sizes':((30,50),(10,10,10)), #add 2 diff network architecture
                 'solver':('adam','sgd')}
    TOPMLP = GridSearchCV(estimator=MLPClassifier(),param_grid=MLPparams)
    TOPMLP.fit(x_train,y_train)
    TOPMLPpredict = TOPMLP.predict(x_test)