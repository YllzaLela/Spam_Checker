#!/usr/bin/env python
# coding: utf-8

# Spam Checker Project made by Yllza Lela  

# Usign Naive Bayes algorithm to classify our data. It is a supervised, classification algorithm.

# The data used here is a dataset taken from the UCI Machine Learning Repository called the "SMS Spam Collection Data Set". 
# It contains one set of SMS messages in English of 5574 messages, tagged accordingly either "ham" (non-spam) or "spam". 
# Can check them out here: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
# 

# # Importing Libraries

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # used for data visualization/plotting etc.
import seaborn as sns #also for data visualization


# # Importing our data

# In[5]:


data = pd.read_csv('data.csv')


# Showing some of the data *Not needed just helps with understanding

# In[7]:


data.head(5) #showing first 5 rows of out data


# In[8]:


data.tail(5) #last 5 datapoints


# Showing number of entries, if there's null elements or not etc. *Not needed just helps with understanding

# In[10]:


data.info() 


# # Visualizing our Data

# In[11]:


ham= data[ data[ 'spam' ] == 0 ] # putting ham (non-spam) messages in one group and showing it


# In[17]:


ham.head(5) # showing the data


# In[14]:


spam= data [ data['spam'] == 1] # same thing with spam messages


# In[16]:


spam.head(5) # showing it


# In[19]:


print ("Spam Percentage= ", (len(spam)/len(data)) * 100, '%' ) # Showing spam percentage out of all data


# In[20]:


print ("Ham Percentage= ", (len(ham)/len(data)) * 100, '%') # showing ham percentage


# In[29]:


sns.countplot(x= data['spam']) 


# # Preparing our Data

# In[30]:


from sklearn.feature_extraction.text import CountVectorizer


# In[39]:


vecto = CountVectorizer() 


# In[40]:


data_countvectorizer = vecto.fit_transform(data['text']) #transforming the 'text' column of our data into numbers


# In[42]:


print(vecto.get_feature_names())  # Optional just helps to understand better


# In[43]:


print (data_countvectorizer.toarray())    


# In[44]:


data_countvectorizer.shape 
# this will show you how may samples there was (5728) and the number of words extracted from those samples (37303)


# # Dividing Data and Training Model

# In[48]:


label = data['spam'].values


# In[47]:


x = data_countvectorizer # the prepared data
y = label # the spam and ham labels 


# In[51]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2) #divided data for 80% training 20% testing
classifier =  MultinomialNB()
classifier.fit(x_train, y_train)  #this is our trained model, all the intelligence happens in here


# # Evaluating our Model

# In[52]:


from sklearn.metrics import classification_report, confusion_matrix


# Creating 2 confusion matrices 1 for train and 1 for test

# In[53]:


y_predict_train = classifier.predict(x_train)
y_predict_train


# In[54]:


a = confusion_matrix(y_train, y_predict_train) # y_train = the truth # y_predict_train= model prediction 
sns.heatmap(a, annot = True ) #visualizing 


# Correctly Classified: 35 and 11 hundred examples
# Missclassified: 20 examples. Bear in mind this is done on training sets. Not testing.
# 

# In[55]:


y_predict_test = classifier.predict(x_test)


# In[56]:


b=confusion_matrix(y_test, y_predict_test)


# In[57]:


sns.heatmap(b, annot= True)


# In[58]:


print(classification_report(y_test, y_predict_test)) #generating a report of our model 

