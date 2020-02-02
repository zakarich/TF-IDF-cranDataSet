#!/usr/bin/env python
# coding: utf-8

# # Evaluation of search query results

# ### Loading the dataset - documents in which the search needs to be done

# In[1]:


cranInp = open('cran.all.1400').read().replace('.T\n','\n').replace('.A\n','\n').replace('.B\n','\n').replace('.W\n','\n').split('\n.I ')

# Figure out what is happening here. You have already seen this.
from sklearn.feature_extraction.text import TfidfVectorizer
Vcount = TfidfVectorizer(analyzer='word', ngram_range=(1,1), stop_words = 'english')
countMatrix = Vcount.fit_transform(cranInp)


# ### Loading the sample queries

# In[2]:


cranQuery = open('cran.qry').read().replace('.W\r','').split('.I ')[1:]

cranQuery[0]
queryDict = dict()
queryVects = dict()

for item in cranQuery:
    stuff = item.split('\r\n\n')
    queryDict[stuff[0]] = stuff[-1].strip('\r\n').replace('\r',' ')
    queryVects[stuff[0]] = Vcount.transform([stuff[-1].strip('\r\n').replace('\r',' ')])


# ### Loading the Query relevance Judgements

# In[3]:


from collections import defaultdict
queryRel = open('cranqrel').read().split('\n')

queryRelDict = defaultdict(dict)
for item in queryRel:
    stuff = item.split()
    try:
        queryRelDict[stuff[0]][stuff[2]].append(stuff[1])
    except:
        queryRelDict[stuff[0]][stuff[2]] = list()
        queryRelDict[stuff[0]][stuff[2]].append(stuff[1])


# In[4]:


print (queryVects)


# In[5]:


# Query 1: "what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft"
from sklearn.metrics.pairwise import cosine_similarity

cosMattf = cosine_similarity(queryVects['001\n.W\nwhat similarity laws must be obeyed when constructing aeroelastic models\nof heated high speed aircraft .\n'],countMatrix)
related_docs_indices = cosMattf[0].argsort()[:-11:-1]


for item in related_docs_indices:
    print ('Document', item+1, cosMattf[0][item])

tp = list()
for item in queryRelDict['1'].keys():
    for stuff in related_docs_indices:
        if str(stuff+1)in queryRelDict['1'][item]:
            tp.append(stuff+1)


# ### Precision

# In[6]:


print (tp)

#All other entries which are in related_docs_indices but not in queryRelDict['1'] are false positives
precision = 1.0*len(tp)/len(related_docs_indices)

print ('Precision is', precision)


# ### Recall

# In[7]:


recallDocLen = 0
for item in queryRelDict['1'].keys():
    recallDocLen += len(queryRelDict['1'][item])
    
print (recallDocLen)

recall = 1.0*len(tp)/recallDocLen

print ('recall is', recall)


# ### Precision and Recall @ K

# In[8]:


# change the value for k
k = 30

cosMattf = cosine_similarity(queryVects['001\n.W\nwhat similarity laws must be obeyed when constructing aeroelastic models\nof heated high speed aircraft .\n'],countMatrix)
related_docs_indices = cosMattf[0].argsort()[:-1*(k+1):-1]

tp = list()
for item in queryRelDict['1'].keys():
    for stuff in related_docs_indices:
        if str(stuff+1)in queryRelDict['1'][item]:
            tp.append(stuff+1)
            
precision = 1.0*len(tp)/len(related_docs_indices)
recall = 1.0*len(tp)/recallDocLen

print ('Precision is', precision)
print ('Recall is', recall)


# In[9]:


#Average Precision @ Key

relOrNot = [0]*k
for item in queryRelDict['1'].keys():
    for i in range(len(related_docs_indices)):
        if str(related_docs_indices[i]+1) in queryRelDict['1'][item]:
            relOrNot[i] = 1       
            
print (relOrNot)
avgPs = list()

for i in range(len(relOrNot)):
    if relOrNot[i] == 1:
        print ('P@',i+1,' : ',sum(relOrNot[:i+1])*1.0/(i+1))
        avgPs.append(sum(relOrNot[:i+1])*1.0/(i+1))

        
print ('Average Prcision @ K for query 1 :', sum(avgPs)/len(avgPs))


# In[ ]:




