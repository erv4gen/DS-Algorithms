
# coding: utf-8

# In[1]:


import tensorflow as tf
import nltk


# In[2]:


from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random
import numpy as np
import pickle
from collections import Counter

# In[3]:


lemmatizer = WordNetLemmatizer()
hm_line = 10000000


# In[7]:


def create_lexicon(pos,neg):
    lexicon = []
    for fi in [pos, neg]:
        with open(fi,'r') as f:
            contents = f.readlines()
            for l in contents[:hm_line]:
                all_words = word_tokenize(l)
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] >50:
            l2.append(w)
    print('Lexicon: ',len(l2))
    return l2


# In[4]:


def sample_handling(sample,lexicon,classification):
    featureset = []
    with open(sample,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_line]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] +=1
            features = list(features)
            featureset.append([features , classification ])

    return featureset


# In[ ]:


def create_featureset_and_labels(pos,neg,test_size=0.1):
    lexicon = create_lexicon(pos,neg)
    featureset = []
    featureset += sample_handling('C:\\Users\\15764\\Documents\\Datasets\\ft_bin_class\\pos.txt', lexicon, [1,0])
    featureset += sample_handling('C:\\Users\\15764\\Documents\\Datasets\\ft_bin_class\\neg.txt', lexicon, [0,1])
    random.shuffle(featureset)

    featureset = np.array(featureset)
    testing_size = int(test_size*len(featureset))
    train_x = list(featureset[:,0][:-testing_size])
    train_y = list(featureset[:,1][:-testing_size])

    test_x = list(featureset[:,0][-testing_size:])
    test_y = list(featureset[:,1][-testing_size:])
    return  train_x, train_y, test_x, test_y
