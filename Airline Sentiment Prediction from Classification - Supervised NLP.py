#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv('../DATA/airline_tweets.csv')


# In[5]:


df.head()


# In[7]:


df['airline_sentiment'].value_counts()


# In[9]:


sns.countplot(data=df, 
              x='airline_sentiment')


# In[15]:


sns.countplot(data=df, 
              x='negativereason')
plt.xticks(rotation=90);


# In[17]:


sns.countplot(data=df, 
              x='airline', 
              hue='airline_sentiment')


# In[19]:


data = df[['airline_sentiment', 'text']]


# In[21]:


data.head()


# In[23]:


X = df['text']
y = data['airline_sentiment']


# In[25]:


## TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[29]:


## VECTORIZATION
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(X_train)
X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# In[33]:


X_test_tfidf
# we do not .todense() these as they are extremely large sets of data


# In[37]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)


# In[39]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_tfidf, y_train)


# In[42]:


from sklearn.svm import SVC, LinearSVC
rbf_svc = SVC()
rbf_svc.fit(X_train_tfidf, y_train)


# In[43]:


linear_svc = LinearSVC()
linear_svc.fit(X_train_tfidf, y_train)


# In[64]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
def report(model, X_test_tfidf, y_test):
    preds = model.predict(X_test_tfidf)
    print(classification_report(y_test, preds))
    ConfusionMatrixDisplay(confusion_matrix(y_test, preds)).plot()


# In[70]:


report(nb, X_test_tfidf, y_test)


# In[72]:


report(linear_svc, X_test_tfidf, y_test)


# In[78]:


from sklearn.pipeline import Pipeline
## this pipeline is fed to raw string tweets
pipe = Pipeline([
    ('tfidf', TfidfVectorizer()), 
    ('svc', LinearSVC())
])

pipe.fit(X, y)


# In[81]:


new_tweet = ['good flight']
pipe.predict(new_tweet)


# In[83]:


new_tweet = ['bad flight']
pipe.predict(new_tweet)


# In[85]:


new_tweet = ['ok flight']
pipe.predict(new_tweet)


# In[ ]:




