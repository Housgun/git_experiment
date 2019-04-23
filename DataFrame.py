# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:59:55 2019

@author: User
"""
#%%
#1101 台泥
#1102 亞泥
#1103 嘉泥
#1104 環泥



#%%
import re
import os
import numpy as np
import pandas as pd
import jeiba

News = []

#%% Data frame
#parser = AP()
#parser.add_argument("file", help = "News file")

#os.chdir("E:\\2019spring\PaperData")
with open("TBMC_news.txt", 'r', encoding  = 'utf-8' ) as file:
    for line in file:
        News.append(line.strip('\n'))


NewsCluster = []
tmp = []
for line in News:
    if(line == ''):
        NewsCluster.append(tmp)
        tmp = []
    else:
        tmp.append(line)


news = []
for line in NewsCluster:
    tmp = [line[2][8:], re.sub('/', '-', line[6][6:]).strip(' ')]
    news.append(tmp)

newspd = pd.DataFrame(news, columns = ['title', 'date'])

num = []
seg= []
change = []
for i in range(len(newspd)):
    num.append([])
    seg.append([])
    change.append([])
    
newspd['stockno'] = pd.Series(num)
newspd['seg'] = pd.Series(seg)
newspd['chg'] = pd.Series(change)

for i in range(len(newspd)):
    if('台泥' in newspd['title'][i]):
        newspd['stockno'][i].append('1101')
    if('亞泥' in newspd['title'][i]):
        newspd['stockno'][i].append('1102')
    if('嘉泥' in newspd['title'][i]):
        newspd['stockno'][i].append('1103')
    if('環泥' in newspd['title'][i]):
        newspd['stockno'][i].append('1104')
    if('水泥' in newspd['title'][i]):
        newspd['stockno'][i].append('666')

#%% Embedding 
#    
#NewsDic = {}
#NewsVec = {}

stopSet = set()
with open("StopWords.txt", "r", encoding = "utf-8") as Sfile:
    for word in Sfile:
        stopSet.add(word.strip('\n'))    
        
#for i in range(len(newspd)):
#    NewsDic[newspd["date"][i]] = []
#    NewsVec[newspd["date"][i]] = []
#    
#summary = []

for i in range(len(newspd)):
    tmp = []
    date = newspd['date'][i]
    text = newspd['title'][i].split(' ')
    for item in text:
        words = jeiba.cut(item, 1, 1)
        for word in words:
            if(word not in stopSet):
                tmp.append(word)
    newspd['seg'][i] = tmp
    
#for i in range(len(summary)):
#    NewsDic[summary[i][0]].append(summary[i][1])


#%% Doc2Vec model

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
from Reporter import reporter


m1101 = reporter(1101)
m1102 = reporter(1102)
m1103 = reporter(1103)
m1104 = reporter(1104)

fakenews = [m1101, m1102,m1103,m1104]

summary = []
for news in fakenews:
    for i, j in news.news.items():
        for n in j:
            summary.append(n)

#summary = []
#with open('Vocabulary.txt', 'r', encoding = 'utf-8') as f:
#    for line in f:
#        summary.append(word_tokenize(line))

corpus = [TaggedDocument(item, [i]) for i, item in enumerate(summary)]
model = Doc2Vec(vec_size = 5, negative = 10, dbow_words = 1)
model.build_vocab(corpus)
#model = Doc2Vec(corpus, vec_size = 5, negative = 10, dbow_words = 1)

max_epochs = 100
vec_size = 100
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(corpus)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(corpus,
                total_examples=model.corpus_count,
                epochs=model.iter)
    model.alpha -= 0.0002
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")


#%%
def label_redo(l):
  for i, j in enumerate(l):
    if((j == 2) and (i != len(l) -1)):
        k = i+1
        while((l[k] == 2) and (k != len(l) -1)):
            k +=1 
        if(k >= len(l)):
            break
        else:
            l[i] = l[k]
  return l





#%% train without 1101
from sklearn import svm

clf = svm.SVC(kernel = 'precomputed')

midsky = [m1102,m1103,m1104]

X_train_text = []
y_train = []

cnt = 0
for fakenews in midsky:
  L = label_redo(fakenews.L)
  text = fakenews.news
  for i, j in enumerate(text.items()):
    if(i+1 == len(text)):
      break
    for rep in j[1]:
      if(L[i+1] == 2):
        break
      X_train_text.append(rep)
      y_train.append(L[i+1])
      cnt += 1
#      if(cnt == 1724):
#        print(i, L[i+1])
      
X_train = []
for i in X_train_text:
  X_train.append(model.infer_vector(i))
  
#%%
X_test = []
Y_test = []

text = m1101.news
L = m1101.L

for i, j in enumerate(text.items()):
    if(i+1 == len(text)):
      break
    for rep in j[1]:
      X_test.append(rep)
      Y_test.append(L[i+1])
      
X = []
for i in X_test:
  X.append(model.infer_vector(i))
  


clf.fit(X_train, y_train)
clf.score(X, Y_test)

#print()


#%% train with 1101
  
#clf = LinearSVC(random_state=0, tol=1e-5)  

#for i in range(1100):
#  X_train.append(X[i])
#  y_train.append(Y_test[i])
#  
#new_x = X[1100:]
#new_y = Y_test[1100:]
#
#
#
#clf.fit(X_train, y_train)
#clf.score(new_x, new_y)






