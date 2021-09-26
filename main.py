import os
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.model_selection
import sklearn.naive_bayes as NB

#finding BBC subdir in local directory (BBC folder needs to be in project folder COMP472_P1)
for subdir, dirs, files in os.walk(os.curdir):
    for dir in dirs:
        if dir == "BBC":
            bbcpath = os.path.join(subdir,dir)
            #print(bbcpath)

#step 2
#TODO insert step 2 code here

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
vectorizerinator = CountVectorizer()
X = vectorizerinator.fit_transform(corpus.data)
print(vectorizerinator.get_feature_names())
print(X.toarray())
