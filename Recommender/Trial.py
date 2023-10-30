#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


tour = pd.read_csv('finaldata.csv')


# In[3]:


tour.head()


# In[4]:


tour.shape


# In[5]:


tourism = tour[['state','City','Place','URL']]


# In[6]:


tourism.head()


# In[7]:


tourism


# In[8]:


tourism.isnull().sum()


# In[9]:


tourism.duplicated().sum()


# In[10]:


tourism.iloc[0].Place


# In[11]:


tourism['Place'] = tourism['Place'].apply(lambda x:x.split())


# In[12]:


tourism['City'] = tourism['City'].apply(lambda x:x.split())


# In[13]:


#tourism['Distance'] = tourism['Distance'].apply(lambda x:x.split())


# In[14]:


#tourism['Place_desc'] = tourism['Place_desc'].apply(lambda x:x.split())


# In[15]:


tourism.head()

print(tourism['Place'].dtype)
# In[16]:


if tourism['Place'].dtype == 'float':
    print("float")
else:
    print("no ")


# In[17]:


def remove_numeric(lst):
    return list(map(lambda x: x if not x.isdigit() else '', lst))

# Apply the function to the 'place' column using the map() function
tourism['Place'] = tourism['Place'].map(remove_numeric)

# Print the updated DataFrame
tourism


# In[18]:


tourism['Place'] = tourism['Place'].apply(lambda x: x[1:])


# In[19]:


tourism


# In[20]:


print(tourism.columns)


# In[21]:


tourism['City'] = tourism['City'].apply(lambda x:[i.replace(" ","") for i in x])
tourism['Place'] = tourism['Place'].apply(lambda x:[i.replace(" ","") for i in x])
#tourism['Best_time_to_visit'] = tourism['Best_time_to_visit'].apply(lambda x:[i.replace(" ","") for i in x])
#tourism['Distance'] = tourism['Distance'].apply(lambda x:[i.replace(" ","") for i in x])
#tourism['Place_desc'] = tourism['Place_desc'].apply(lambda x:[i.replace(" ","") for i in x])


# In[22]:


tourism


# In[23]:


tourism['tags'] = tourism['City'] + tourism['Place'] 


# In[24]:


tourism.head()


# In[25]:


tourism['tags'].head()


# In[26]:


new_tour = tourism[['state', 'City' ,'Place','URL','tags']]


# In[27]:


new_tour


# In[28]:


new_tour['City'] = new_tour['City'].apply(lambda x:" ".join(x))


# In[29]:


new_tour['Place'] = new_tour['Place'].apply(lambda x:" ".join(x))


# In[30]:


new_tour['tags'] = new_tour['tags'].apply(lambda x:" ".join(x))


# In[31]:


new_tour


# In[32]:


new_tour.to_csv("trialfinal.csv", index = False)


# In[33]:


import nltk


# In[34]:


from nltk.stem.porter import PorterStemmer
ps  = PorterStemmer()


# In[35]:


def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[36]:


new_tour['tags'] = new_tour['tags'].apply(stem)


# In[37]:


new_tour['tags'][0]


# In[38]:


new_tour['tags'] = new_tour['tags'].apply(lambda x:x.lower())


# In[39]:


new_tour['tags'][0]


# In[40]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000 , stop_words = 'english')


# In[41]:


vectors = cv.fit_transform(new_tour['tags']).toarray()


# In[42]:


vectors


# In[43]:


vectors[0]


# In[44]:


cv.get_feature_names_out()


# In[45]:


ps.stem('manali captur the sceneri of old manali')


# In[46]:


from sklearn.metrics.pairwise import cosine_similarity


# In[47]:


similarity = cosine_similarity(vectors)


# In[48]:


sorted(list(enumerate(similarity[0])),reverse = True,key = lambda x:x[1])[1:6]


# In[49]:


new_tour


# In[50]:


def recommend(tourism):
    city_match = new_tour[new_tour['City'] == tourism]
    state_match = new_tour[new_tour['state'] == tourism]
    
    if not city_match.empty:
        tour_index = city_match.index[0]
    elif not state_match.empty:
        tour_index = state_match.index[0]
    else:
        print("No matching city or state found.")
        return
    
    distances = similarity[tour_index]
    tour_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    #data = []
    for i in tour_list:
        print(new_tour.iloc[i[0]].Place, new_tour.iloc[i[0]].City , new_tour.iloc[i[0]].state , new_tour.iloc[i[0]].URL)
        #data.append(this_list)
        
    
    return 


# In[51]:


recommend('Nainital')


# In[52]:


def recommend(tourism):
    city_match = new_tour[new_tour['City'] == tourism]
    state_match = new_tour[new_tour['state'] == tourism]
    
    if not city_match.empty:
        tour_index = city_match.index[0]
    elif not state_match.empty:
        tour_index = state_match.index[0]
    else:
        print("No matching city or state found.")
        return
    
    distances = similarity[tour_index]
    tour_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    data = []
    for i in tour_list:
        final_list =[(new_tour.iloc[i[0]].Place),(new_tour.iloc[i[0]].City),(new_tour.iloc[i[0]].state),(new_tour.iloc[i[0]].URL)]
         
        data.append(final_list)
        
    
    return data


# In[58]:


recommend('Nainital')


# In[54]:


import pickle


# In[55]:


pickle.dump(new_tour,open('new_tour.pkl','wb'))


# In[56]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[57]:


pickle.dump(tour,open('tour.pkl','wb'))


# In[ ]:




