import numpy as np
import matplotlib.pyplot as plt
import pandas
import sklearn
import pandas as pd

# read drug200.csv file
drug200read = pd.read_csv("drug200.csv")

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
