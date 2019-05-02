#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from sklearn import ensemble


# In[ ]:


df = pd.read_csv('train_ver2.csv')


# # Checking the total number of missing values for each attribute

# In[ ]:


df.isnull().sum()


# # Dropping the attributes having more than a million missing data 

# In[ ]:



df = df.drop(["ult_fec_cli_1t", "conyuemp"], axis=1) 


# # Random Forest 

# In[ ]:




predic_custid = defaultdict(list)
cust_id = df_train['ncodpers'].values
for c in df_train.columns:
    if c != 'ncodpers':
        # Adding the column to y for training
        y_train = df_train[c]
        # Adding all the other columns to x for training
        x_train = df_train.drop([c, 'ncodpers'], 1)
        
        #Random Forest
        clf = ensemble.RandomForestClassifier(n_estimators=60, n_jobs=-1,max_depth=10, min_samples_split=10, verbose=1)

        clf.fit(x_train, y_train)
        #Training the classifier
        mod = clf.predict_proba(x_train)[:,1]
      
        
        for id, p in zip(cust_id, mod):
            predic_custid[id].append(p)
            
        print(roc_auc_score(y_train, mod))

    
train_preds = {}
for id, p in predic_custid.items():
    preds = [i[0] for i in sorted([i for i in zip(df_train.columns[1:], p) if i[0] not in already_active[id]], key=lambda i:i [1], reverse=True)[:7]]
    train_preds[id] = preds
    
test_preds = []
for row in sample.values:
    id = row[0]
    p = train_preds[id]
    test_preds.append(' '.join(p))
    
sample['added_products'] = test_preds

sample.to_csv('RFC_sub.csv', index=False)

