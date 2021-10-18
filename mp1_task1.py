import numpy as np
import os
import os.path as ospath
from glob import glob
from zipfile import ZipFile

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import sklearn.naive_bayes as skNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# === STEP 1 ===
print("step 1 :: in progress . . .")
#finding BBC subdir in local directory (BBC folder needs to be in project folder COMP472_P1)
bbcpath = ospath.join(os.curdir, 'BBC')
if not os.path.isdir('BBC'):
    with ZipFile('BBC-20210914T194535Z-001.zip','r') as zipObj:
        zipObj.extractall()
        print("-> extracted BBC zip")
else:
    print("-> BBC dir exists")
print("step 1 :: done")

# === STEP 2 ===
print("step 2 :: in progress . . .")
categories = list()                                 #list of class names
fcount = [0]                                        #list of file count in each class
for path in glob(bbcpath+'\*'):
    if ospath.isdir(path) and path != bbcpath:
        categories.append(ospath.basename(path))    #add folder name as category
        fcount[0] = 0                               #re-initialize file count to 0
        for file in glob(path+'\*.txt'):
            fcount[0] = fcount[0] + 1               #count files in folder
        fcount.append(fcount[0])                    #add final count to fcount
fcount.pop(0)                                       #pop fcount[0] (file counter)

plt.bar(x = np.array(categories),height = np.array(fcount))
plt.xlabel("Categories")
plt.ylabel("Number of Articles")
plt.title("Distribution of Articles by Category")
plt.savefig('bbc-distribution.pdf')
#plt.show()
print("step 2 :: done")

# === STEP 3 ===
print("step 3 :: in progress . . .")
corpus = datasets.load_files(container_path = bbcpath, description = "BBC dataset", encoding = 'latin1', shuffle=False)
# corpus is a 'Bunch' with attributes data[], target[], target_names[], DESCR, filenames[]
#           data[] = content of file as one string
#         target[] = class of the file as int (index for target_name)
#   target_names[] = name of class
#            DESCR = description of dataset
#      filenames[] = file names
# -> if we need to randomize data, set shuffle to True(default)
print("step 3 :: done")

# === STEP 4 ===
print("step 4 :: in progress . . .")
vectorizerinator = CountVectorizer()
X = vectorizerinator.fit_transform(corpus.data)
vocabulary = vectorizerinator.get_feature_names()
distribution = X.toarray()
# distribution is an array where distribution[doc][word] = wordcount
print("step 4 :: done")

# === STEP 5 ===
print("step 5 :: in progress . . .")
dist_train, dist_test, target_train, target_test = train_test_split(distribution, corpus.target, random_state=None)
print("step 5 :: done")

# === STEP 6 ===
print("step 6 :: in progress . . .")
mnbc = skNB.MultinomialNB()
mnbc.fit(dist_train, target_train)
pred = mnbc.predict(dist_test)
answ = target_test
print ("step 6 :: done")

# === STEP 7 ===
print("step 7 :: in progress . . .")

if os.path.exists("bbc-performance.txt"):
  os.remove("bbc-performance.txt")

def save_results(title = None, version = None):
    with open('bbc-performance.txt','a') as f:
        #===================================================================================
        f.writelines( "(a) :: ============================================\n"
                    + "        {}{}\n".format("MultinomialNB" if title == None else title, "" if version == None else ", try " + str(version))
                    + "       ============================================\n")
        #===================================================================================
        cfs_matrix = confusion_matrix(y_pred = pred, y_true = answ)
        f.writelines( "\n(b) :: confusion matrix\n\n"
                    + np.array2string(cfs_matrix) + "\n")
        #===================================================================================
        clssf_rep = classification_report(y_pred = pred, y_true = answ, target_names = corpus.target_names)
        f.writelines( "\n(c) :: per class precision, recall and F1-measure\n\n"
                    + clssf_rep)
        #===================================================================================
        f.writelines( "\n(d) :: accuracy, macro-average F1 and weighted-average F1\n"
                    + "{:>15} : {:.6f}\n".format("Accuracy", accuracy_score(y_pred= pred, y_true = answ))
                    + "{:>15} : {:.6f}\n".format("Macro-F1", f1_score(y_pred= pred, y_true = answ, average = 'macro'))
                    + "{:>15} : {:.6f}\n".format("Weighted-F1", f1_score(y_pred= pred, y_true = answ, average = 'weighted')))
        #===================================================================================
        f.writelines( "\n(e) :: per class prior probabilities\n")
        for c in range(len(categories)):
            f.writelines( "{:>15} : {:.6f} \n".format(categories[c], float(fcount[c]/sum(fcount))))
        #===================================================================================
        f.writelines( "\n(f) :: size of vocabulary\n"
                    + "{:>16} words\n".format(len(vocabulary)))
        #===================================================================================
        f.writelines( "\n(g) :: # of word-tokens per class\n")
        last, next = 0, 0
        for c in range(len(categories)):
            last = next
            next = next + fcount[c]
            f.writelines("{:>15} : {:5n} words\n".format(categories[c], X[last:next].sum()))
        #===================================================================================
        f.writelines( "\n(h) :: # of word-tokens in corpus\n"
                    + "{:>16} words\n".format(X.sum()))
        #===================================================================================
        f.writelines( "\n(i) :: # and % of words with frequency of 0 per class\n")
        last, next = 0, 0
        for c in range(len(categories)):
            last = next
            next = next + fcount[c]
            wcount = np.transpose(X[last:next].sum(axis=0))
            wf0 = 0
            for w in wcount :
                if w == 0:
                    wf0 = wf0 + 1
            f.writelines("{:>15} : {:5n} words ({:.2%})\n".format(categories[c], wf0, wf0/len(vocabulary)))
        #===================================================================================
        wcount = np.transpose(X.sum(axis=0))
        wf1 = 0
        for w in wcount :
                if w == 1:
                    wf1 = wf1 + 1
        f.writelines( "\n(j) :: # and % of words with frequency of 1 in corpus\n"
                    + "{:>16} words ({:.2%})\n".format(wf1, wf1/len(vocabulary)))
        #===================================================================================
        f.writelines( "\n(k) :: log prob of 2 favorite words\n"
                    + "\n")    
        for x in range(5):
            f.writelines("feature {vocab} for category {cat} is {prob_log} \n".format(vocab = vocabulary[9999], cat = categories[x],prob_log = mnbc.feature_log_prob_[x][9999]))
        
        f.writelines("\n------------------------------------------\n")

        for x in range(5):
            f.writelines("feature {vocab} for category {cat} is {prob_log} \n".format(vocab = vocabulary[29000], cat = categories[x],prob_log = mnbc.feature_log_prob_[x][29000]))

    
        f.writelines("\n")

    f.close()
save_results("MultinomialNB default values", 1)





print("step 7 :: done")

# === STEP 8 ===
print("step 8 :: in progress . . .")

mnbc = skNB.MultinomialNB()
mnbc.fit(dist_train, target_train)
pred = mnbc.predict(dist_test)
save_results("MultinomialNB default values", 2)

print("step 8 :: done")

# === STEP 9 ===
print("step 9 :: in progress . . .")

mnbc = skNB.MultinomialNB(alpha=0.0001)
mnbc.fit(dist_train, target_train)
pred = mnbc.predict(dist_test)
save_results("MultinomialNB smoothing value = 0.0001")

print("step 9 :: done")

# === STEP 10 ===
print("step 10 :: in progress . . .")

mnbc = skNB.MultinomialNB(alpha=0.9)
mnbc.fit(dist_train, target_train)
pred = mnbc.predict(dist_test)
save_results("MultinomialNB smoothing value = 0.9")

print("step 10 :: done")
