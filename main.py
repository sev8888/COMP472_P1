import numpy as np
import os
import os.path as ospath
from glob import glob

from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import sklearn.naive_bayes as skNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#finding BBC subdir in local directory (BBC folder needs to be in project folder COMP472_P1)
for subdir, dirs, files in os.walk(os.curdir):
    for dir in dirs:
        if dir == "BBC":
            bbcpath = ospath.join(subdir,dir)

# === STEP 2 ===
categories = list()
fcount = [0]
for path in glob(bbcpath+'\*'):
    if ospath.isdir(path) and path != bbcpath: #ignore source folder BBC
        categories.append(ospath.basename(path))
        fcount[0] = 0
        for file in glob(path+'\*.txt'):
            fcount[0] = fcount[0] + 1 #fcount[0] serves as our counter
        fcount.append(fcount[0])
fcount.pop(0) #remove counter fcount[0]

#TODO finish step 2 code here


# === STEP 3 ===
corpus = datasets.load_files(container_path = bbcpath, description = "BBC dataset", encoding = 'latin1', shuffle=False)
# returns a 'Bunch' with attributes data[], target[], target_names[], DESCR, filenames[]
#           data[] = content of file as one string
#         target[] = class of the file as int (index for target_name)
#   target_names[] = name of class
#            DESCR = description of dataset
#      filenames[] = file names
# -> if we need to randomize data, set shuffle to True(default)

# === STEP 4 ===
vectorizerinator = CountVectorizer()
X = vectorizerinator.fit_transform(corpus.data)
dictionary = vectorizerinator.get_feature_names()
distribution = X.toarray() # dist[doc][word] = wordcount

# === STEP 5 ===
dist_train, dist_test, target_train, target_test = train_test_split(distribution, corpus.target, random_state=None)

# === STEP 6 ===
mnbc = skNB.MultinomialNB()
mnbc.fit(dist_train, target_train)
pred = mnbc.predict(dist_test)
answ = target_test

# === STEP 7 ===
n = 1

with open('bbc-performance.txt','a') as f:
    f.writelines("\n(a) :: ===== Multi-nomialNB default values, try "+n+" ====")
    f.writelines("\n(b) :: \n"+confusion_matrix(y_pred = pred, y_true = answ))
    f.writelines("\n(b) :: \n"+classification_report(y_pred = pred, y_true = answ, target_names = corpus.target_names))