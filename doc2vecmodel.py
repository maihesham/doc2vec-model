import pandas as pd
import numpy as np
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import glob
import re
from sklearn.cross_validation import train_test_split
import nltk
from nltk.corpus import stopwords
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) <= 2:
                continue
            tokens.append(word.lower())
    return tokens
#to read from doc
def readFilepos(Filename):
        f = open(Filename,"r")
        lines = f.read()
        lines=lines.lower()
        lines=re.sub('[^A-Za-z]+',' ',  lines)
        cleanpostive2.append(lines)
def readFileneg(Filename):
        f = open(Filename,"r")
        lines = f.read()
        lines=lines.lower()
        lines=re.sub('[^A-Za-z]+',' ',  lines)
        cleannegtive2.append(lines)
def getAllwords():
    for t in negativetexts:
          readFileneg(t)
    for t in postivetexts:
          readFilepos(t) 
def makeFeatures():
    for i in cleannegtive2:
        feat.append(i)
        goals.append("neg")
    for i in cleanpostive2:
        feat.append(i)
        goals.append("pos")  
def makeTagges():
    i=0
    j=0
    while i< len(train_set):
        taggedtrain.append(TaggedDocument(tokenize_text(train_set[i]),goal_train[i]))
        i=i+1
    while j < len(test_set):
        taggedtest.append(TaggedDocument(tokenize_text(test_set[j]),goal_test[j]))
        j=j+1 
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs
    targets, regressors = zip(*[(doc.tags, model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors  
def makenodemazation(do):
    maxx=max(do)
    minn=min(do)
    X=[]
    for i in do:
        n = (i-minn)/(maxx-minn)
        X.append(n)
    return X       
############################################################################################################start          
postivetexts=glob.glob("C:/Users/egypt2/Desktop/review_polarity/txt_sentoken/pos/*.txt") #names of all pos texts
negativetexts=glob.glob("C:/Users/egypt2/Desktop/review_polarity/txt_sentoken/neg/*.txt") #names of all neg texts          
cleannegtive2=[]
cleanpostive2=[]
feat=[]
goals=[]
getAllwords()
makeFeatures()
train_set, test_set, goal_train, goal_test = train_test_split(feat,goals,train_size =0.7,random_state=1)
taggedtrain=[]
taggedtest=[]
makeTagges()
new_model = gensim.models.Doc2Vec(vector_siz=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025)
new_model.build_vocab(taggedtrain)
for epoch in range(30):
    new_model.train(utils.shuffle(taggedtrain), total_examples=len(taggedtrain), epochs=1)
    new_model.alpha =new_model.alpha- 0.002
    new_model.min_alpha = new_model.alpha
y_train, Xtrain = vec_for_learning(new_model,taggedtrain)
X_train=[]
for i in Xtrain:
    X_train.append(makenodemazation(i))
y_test, Xtest = vec_for_learning(new_model, taggedtest)
X_test=[]
for i in Xtest:
    X_test.append(makenodemazation(i))
##########################################################################################################logistic
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import cross_val_score


def randomInt (low , high) :
    return random.randint(low,high)

def randomFloat (low , high):
    return random.uniform(low, high)
from sklearn.linear_model import LogisticRegression
itr = randomInt(100,1000)
t = randomFloat(0000.1,0.01)
logistic = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=itr, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=0, solver='liblinear', tol=t,
          verbose=0, warm_start=False)
logistic.fit( X_train,y_train)
from sklearn.metrics import confusion_matrix
pred=logistic.predict(X_test)
def accuracy(goal_test,goal_predict):
    correct = 0
    for i,j in zip(goal_test,goal_predict) :  
        if i == j:
            correct += 1
    accuracy = float(correct)/len(goal_test)  #accuracy 
    return accuracy
print("Logistic_Regression algorithm accuracy is : %f" %(accuracy(y_test,pred)))
#########################################################predict
def readFilepredict1(Filename):
        f = open(Filename,"r")
        lines = f.read()
        lines=lines.lower()
        lines=re.sub('[^A-Za-z]+',' ',  lines)
        testfile.append(lines)
testfile=[]
readFilepredict1("C:/Users/egypt2/Desktop/review_polarity/txt_sentoken/pos.txt")
new=""
for i in testfile:
    new+=i
Y=tokenize_text(new)
X=new_model.infer_vector(Y, steps=20)
XX=makenodemazation(X)
ih=[]
ih.append(list(XX))
predtext=logistic.predict(ih)
print("test is ",predtext)