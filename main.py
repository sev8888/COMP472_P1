import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from sklearn.model_selection import  train_test_split

# -----------
# PART 2
# -----------

# read drug200.csv file
drug200read = pd.read_csv("drug200.csv")

# -----------
# PART 3
# -----------
# storing all the Drug value from Drug colume to the DrugList
DrugList = drug200read[["Drug"]].values

# create 5 differences countable variable for 5 different drugs
drugCountA = 0
drugCountB = 0
drugCountC = 0
drugCountX = 0
drugCountY = 0

# Accessing each values in DrugList and count its total number

for drug in DrugList:
    if(drug == "drugA"):
        drugCountA += 1
    if(drug == "drugB"):
        drugCountB += 1
    if(drug == "drugC"):
        drugCountC += 1
    if(drug == "drugX"):
        drugCountX += 1
    if(drug == "drugY"):
        drugCountY += 1

# print the total number of value for each drug on the Python Console
print("Drug A: " + str(drugCountA))
print("Drug B: " + str(drugCountB))
print("Drug C: " + str(drugCountC))
print("Drug X: " + str(drugCountX))
print("Drug Y: " + str(drugCountY))

# plot the drug-distribution on the bar chart
drugData = [drugCountA,drugCountB,drugCountC,drugCountX,drugCountY]
drugNames = ["DrugA", "DrugB", "DrugC", "DrugX", "DrugY"]
plt.bar(drugNames,drugData)
plt.savefig('drug-distribution.pdf')

# -----------
# PART 4
# -----------
# Convert BP to numerical number
drug200read["BP"] = pd.Categorical(drug200read["BP"],['LOW','NORMAL','HIGH'],ordered=True)
drug200read["BP"] = drug200read["BP"].cat.codes

# Convert Cholesterol to numerical number
drug200read["Cholesterol"] = pd.Categorical(drug200read["Cholesterol"],['LOW','NORMAL','HIGH'],ordered=True)
drug200read["Cholesterol"] = drug200read["Cholesterol"].cat.codes

# convert Drug to numerical number
# DrugA = 0, DrugB = 1, DrugC = 2, DrugX = 3, DrugY = 4
drug200read["Drug"] = pd.Categorical(drug200read["Drug"],['drugA','drugB','drugC','drugX','drugY'],ordered=True)
drug200read["Drug"] = drug200read["Drug"].cat.codes

# convert Sex to numerical number
drug200read["Sex"] = pd.Categorical(drug200read["Sex"],['M','F'])
drug200read["Sex"] = drug200read["Sex"].cat.codes
drug200read.dropna(axis=0,how='any',inplace=True)
pd.get_dummies(drug200read)

# -----------
# PART 5
# -----------
# Split the dataset using train_test_split
x = drug200read[['Age','Sex','BP','Cholesterol','Na_to_K']]
y = drug200read['Drug']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=10)
print("x_test:")
print(x_test)
print("y_test:")
print(y_test)
