import os
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.model_selection
import sklearn.naive_bayes as NB
import glob

#finding BBC subdir in local directory (BBC folder needs to be in project folder COMP472_P1)
for subdir, dirs, files in os.walk(os.curdir):
    for dir in dirs:
        if dir == "BBC":
            bbcpath = os.path.join(subdir,dir)
            #print(bbcpath)

#step 2
#TODO insert step 2 code here
categories = list()
fcount = [0]
for path in glob.glob(bbcpath+'\*'):
    if os.path.isdir(path) and path != bbcpath: #ignore source folder BBC
        categories.append(os.path.basename(path))
        fcount[0] = 0
        for file in glob.glob(path+'\*.txt'):
            fcount[0] = fcount[0] + 1 #fcount[0] serves as our counter
        fcount.append(fcount[0])
fcount.pop(0) #remove counter fcount[0]
print(categories)
print(fcount)


#step 3
corpus = datasets.load_files(container_path = bbcpath, description = "BBC dataset", encoding = 'latin1', shuffle=False)
# returns a 'Bunch' with attributes data[], target[], target_names[], DESCR, filenames[]
#           data[] = content of file as one string
#         target[] = class of the file as int (index for target_name)
#   target_names[] = name of class
#            DESCR = description of dataset
#      filenames[] = file names
# -> if we need to randomize data, set shuffle to True(default)

#step 4 TODO adapt for step 6
"""vectorizerinator = CountVectorizer()
X = vectorizerinator.fit_transform(corpus.data)
print(vectorizerinator.get_feature_names())
print(X.toarray())"""
