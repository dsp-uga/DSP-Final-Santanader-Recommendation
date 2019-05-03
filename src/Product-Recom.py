#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import accuracy_score
import xgboost as xgb
from io import BytesIO


# In[2]:


train = pd.read_csv("gs://chatrath/train_ver2.csv")
test = pd.read_csv("gs://chatrath/test_ver2.csv")


# In[3]:


train.fecha_dato = pd.to_datetime(train.fecha_dato, format="%Y-%m-%d")
train.fecha_alta = pd.to_datetime(train.fecha_alta, format="%Y-%m-%d")
test.fecha_dato = pd.to_datetime(test.fecha_dato, format="%Y-%m-%d")
test.fecha_alta = pd.to_datetime(test.fecha_alta, format="%Y-%m-%d")


# In[4]:


months = train.fecha_dato.unique()


# In[5]:


train_final = pd.DataFrame()


# In[6]:


#Start with the second month because we need a previous month to compare data with.
i = 1
while i < len(months):
    #Subset all of the data of the new month, which will be compared to the previous month.
    train_new_month = train[train.fecha_dato == months[i]]
    train_previous_month = train[train.fecha_dato == months[i-1]]
    
    print("Original length of train1: ", len(train_new_month))
    print("Original length of train2: ", len(train_previous_month))
    print
    
    #Only select the customers who have data in each month.
    train_new_month = train_new_month.loc[train_new_month['ncodpers'].isin(train_previous_month.ncodpers)]
    train_previous_month = train_previous_month.loc[train_previous_month['ncodpers'].isin(train_new_month.ncodpers)]
    
    print("New length of train_new_month: ", len(train_new_month))
    print("New length of train_previous_month: ", len(train_previous_month))
    print
    
    #Sort by ncodpers (Customer code) to allow for easy subtraction between dataframes later.
    train_new_month.sort_values(by = 'ncodpers', inplace = True)
    train_previous_month.sort_values(by = 'ncodpers', inplace = True)
    
    #These are all of the target features.
    target_cols_all = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1',
                'ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1',
                'ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_dela_fin_ult1','ind_deme_fin_ult1',
                'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_nom_pens_ult1',
                'ind_nomina_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1',
                'ind_recibo_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1']
    
    #Select only the target columns.
    train_new_month_targets = train_new_month[target_cols_all]
    #Add ncodpers to the dataframe.
    train_new_month_targets['ncodpers'] = train_new_month.ncodpers
    #Remove the index.
    train_new_month_targets.reset_index(drop = True, inplace = True)

    #Select only the target columns.
    train_previous_month_targets = train_previous_month[target_cols_all]
    #Add ncodpers to the dataframe.
    train_previous_month_targets['ncodpers'] = train_previous_month.ncodpers
    #Set ncodpers' values to 0, so that there is no effect to this feature when this dataframe is 
    #subtracted from train_new_month_targets.
    train_previous_month_targets.ncodpers = 0
    #Remove the index.
    train_previous_month_targets.reset_index(drop = True, inplace = True)
    
    #Subtract the previous month from the current to find which new products the customers have.
    train_new_products = train_new_month_targets.subtract(train_previous_month_targets)
    #Values will be negative if the customer no longer has a product that they once did. 
    #Set these negative values to 0.
    train_new_products[train_new_products < 0] = 0
    print("Quantity of features to use:")
    #Sum columns to learn about the quantity of the types of new products.
    print(train_new_products.sum(axis = 0))
    
    train_new_products = train_new_products.fillna(0)
    
    #Merge the target features with the data we will use to train the model.
    train_new_products = train_new_products.merge(train_new_month.ix[:,0:24], on = 'ncodpers')
    
    #Add each month's data to the same dataframe.
    train_final = pd.concat([train_final,train_new_products], axis = 0)
    
    print("Length of new dataframe:", len(train_final))
    print
    percent_finished = float(i/len(months))
    print("Percent finished:", percent_finished)
    
    i += 1





# Only select the rows in the dataframe where there is a new product, i.e. where at least one target feature has a value of 1. 

# In[8]:


train_final = train_final.loc[(train_final.iloc[:,0:24] != 0).any(axis=1)]



# In[10]:


len(train_final)


# We need the data from May 2016 because we are only interested in building a model about which new products customers will have in June 2016. Therefore, we need to compare the model's prediction of reccommended products, versus the products the customer already has.

# In[11]:


final_month = train[train.fecha_dato == '2016-05-28']


# In[12]:


df = pd.concat([train_final,test], axis = 0, ignore_index = True)


# In[15]:


df.isnull().sum()


# In[16]:


#Clean the data
print("Step 1/13")
badRows = df[df.ind_empleado.isnull()].index

print("Step 2/13")
df = df.drop(badRows, axis = 0)

print("Step 3/13")
df.canal_entrada = df.canal_entrada.fillna("Unknown")

print("Step 4/13")
df = df.drop("cod_prov", 1)

print("Step 5/13")
df.conyuemp = df.conyuemp.fillna("Unknown")

print("Step 6/13")
df.loc[df.indrel_1mes == '1', 'indrel_1mes'] = 'Primary'
df.loc[df.indrel_1mes == '1.0', 'indrel_1mes'] = 'Primary'
df.loc[df.indrel_1mes == 1, 'indrel_1mes'] = 'Primary'
df.loc[df.indrel_1mes == 1.0, 'indrel_1mes'] = 'Primary'
df.loc[df.indrel_1mes == '2', 'indrel_1mes'] = 'Co-owner'
df.loc[df.indrel_1mes == '2.0', 'indrel_1mes'] = 'Co-owner'
df.loc[df.indrel_1mes == 2, 'indrel_1mes'] = 'Co-owner'
df.loc[df.indrel_1mes == 2.0, 'indrel_1mes'] = 'Co-owner'
df.loc[df.indrel_1mes == '3', 'indrel_1mes'] = 'Former Primary'
df.loc[df.indrel_1mes == '3.0', 'indrel_1mes'] = 'Former Primary'
df.loc[df.indrel_1mes == 3, 'indrel_1mes'] = 'Former Primary'
df.loc[df.indrel_1mes == 3.0, 'indrel_1mes'] = 'Former Primary'
df.loc[df.indrel_1mes == '4', 'indrel_1mes'] = 'Former Co-owner'
df.loc[df.indrel_1mes == '4.0', 'indrel_1mes'] = 'Former Co-owner'
df.loc[df.indrel_1mes == 4, 'indrel_1mes'] = 'Former Co-owner'
df.loc[df.indrel_1mes == 4.0, 'indrel_1mes'] = 'Former Co-owner'

df.indrel_1mes = df.indrel_1mes.fillna('Primary')

print("Step 7/13")
df.nomprov = df.nomprov.fillna("MADRID")

print("Step 8/13")
df.loc[df.renta == '         NA',"renta"] = 0
df.renta = df.renta.astype(float)
df.loc[df.renta == 0, 'renta'] = df[df.renta > 0].groupby('nomprov').renta.transform('median')
df.loc[df.renta.isnull(), "renta"] = df.groupby('nomprov').renta.transform('median')

print("Step 9/13")
df.segmento = df[df.renta <= 98000].segmento.fillna("03 - UNIVERSITARIO")
df.segmento = df[df.renta <= 125500].segmento.fillna("02 - PARTICULARES")
df.segmento = df.segmento.fillna("01 - TOP")

print("Step 10/13")
df.sexo = df.sexo.fillna("V")

print("Step 11/13")
df.tiprel_1mes = df.tiprel_1mes.fillna('I')

print("Step 12/13")
df.ind_nomina_ult1 = df.ind_nomina_ult1.fillna(0.0)
df.ind_nom_pens_ult1 = df.ind_nom_pens_ult1.fillna(0.0)

df['antiguedad'] = df['antiguedad'].astype(int)

print("Step 13/13")
df.loc[df.antiguedad == -999999, 'antiguedad'] = df[df.antiguedad >= 0].antiguedad.median()


# In[17]:


#Feature Engineering
print("Step 1/10")
df.age = df.age.astype(int)
df.loc[df.age < 20,"age"] = 19
df.loc[df.age > 90,"age"] = 91

print("Step 2/10")
df['ageGroup'] = (df.age // 10) - 1

print("Step 3/10")
df['isSpanish'] = df.pais_residencia.map(lambda x: 1 if x == "ES" else 0)

print("Step 4/10")
df['majorCity'] = df.nomprov.map(lambda x: 1 if x == "MADRID" or x == "BARCELONA" else 0)

print("Step 5/10")
df['fecha_alta_year'] = pd.DatetimeIndex(df.fecha_alta).year - 1995
df['fecha_dato_year'] = pd.DatetimeIndex(df.fecha_dato).year - 2015
df['fecha_alta_month'] = pd.DatetimeIndex(df.fecha_alta).month - 1
df['fecha_dato_month'] = pd.DatetimeIndex(df.fecha_dato).month - 1

print("Step 6/10")
df.antiguedad = df.antiguedad.astype(int)
df['antiguedad_years'] = df.antiguedad // 12

print("Step 7/10")
df.loc[df.indrel == 99.0, "indrel"] = 0.0

print("Step 8/10")
df['HAS_ult_fec_cli_1t'] = df.ult_fec_cli_1t.map(lambda x: 1 if(x) else 0)

print("Step 9/10")
df = df.drop('ult_fec_cli_1t', 1)

print("Step 10/10")
df['rentaGroup'] = df.renta.astype(float) // 50000
df.loc[df.renta >= 1000000, "rentaGroup"] = 20
df.loc[df.renta >= 10000000, "rentaGroup"] = 21


# In[18]:


train_final_length = len(train_final) - len(badRows)


# In[19]:


train_final, test = df[:train_final_length], df[train_final_length:]


# In[20]:


print(len(train_final))
print(len(test))


# In[21]:


train_final_training_cols = train_final
train_final_training_cols = train_final_training_cols.drop(target_cols_all, axis=1)
test = test.drop(target_cols_all, axis=1)


# In[22]:


df = pd.concat([train_final_training_cols, test], axis = 0)


# Some features need to be converted to integers with cat.codes. Some of these will then have dummy variables created from them, however features such as pais_residencia will not, because too many features would be created (there are over 100 countries in this dataset).

# In[23]:


print("Step 1/6")
df.pais_residencia = df.pais_residencia.astype('category').cat.codes
print("Step 2/6")
df.canal_entrada = df.canal_entrada.astype('category').cat.codes
print("Step 3/6")
df.nomprov = df.nomprov.astype('category').cat.codes
print("Step 4/6")
final_month.nomprov = final_month.indrel_1mes.astype('category').cat.codes
print("Step 5/6")
df = pd.get_dummies(df, columns = ['ind_empleado','sexo','tiprel_1mes','indresi',
                                   'indext','conyuemp','indfall','segmento','indrel_1mes'])
print("Step 6/6")
#Drop the date features because we can't use them to train the model.
df = df.drop(['fecha_dato', 'fecha_alta'], axis = 1)


# In[24]:


train_final_training_cols, test = df[:train_final_length], df[train_final_length:]


# In[25]:


print("Step 1/11")
#Get the target columns
labels = train_final[target_cols_all]

print("Step 2/11")
#Add ncodpers to the dataframe
labels['ncodpers'] = train_final.ncodpers

print("Step 3/11")
labels = labels.set_index("ncodpers")

print("Step 4/11")
stacked_labels = labels.stack()

print("Step 5/11")
filtered_labels = stacked_labels.reset_index()

print("Step 6/11")
filtered_labels.columns = ["ncodpers", "product", "newly_added"]

print("Step 7/11")
#Only select the rows where there are a new product.
filtered_labels = filtered_labels[filtered_labels["newly_added"] == 1]

print("Step 8/11")
#Merge with the training features.
multiclass_train = filtered_labels.merge(train_final_training_cols, on="ncodpers", how="left")

print("Step 9/11")
train_final = multiclass_train

print("Step 10/11")
labels_final = train_final['product']

print("Step 11/11")
train_final_ncodpers = train_final.ncodpers
#Remove the columns that are not needed to train the model.
train_final = train_final.drop(['ncodpers','newly_added','product'], axis = 1)


# In[26]:


pd.set_option('display.max_columns', 60)


# In[27]:


#Clean the data
print("Step 1/13")
badRows = final_month[final_month.ind_empleado.isnull()].index

print("Step 2/13")
final_month = final_month.drop(badRows, axis = 0)

print("Step 3/13")
final_month.canal_entrada = final_month.canal_entrada.fillna("Unknown")

print("Step 4/13")
final_month = final_month.drop("cod_prov", 1)

print("Step 5/13")
final_month.conyuemp = final_month.conyuemp.fillna("Unknown")

print("Step 6/13")
final_month.loc[final_month.indrel_1mes == '1', 'indrel_1mes'] = 'Primary'
final_month.loc[final_month.indrel_1mes == '1.0', 'indrel_1mes'] = 'Primary'
final_month.loc[final_month.indrel_1mes == 1, 'indrel_1mes'] = 'Primary'
final_month.loc[final_month.indrel_1mes == 1.0, 'indrel_1mes'] = 'Primary'
final_month.loc[final_month.indrel_1mes == '2', 'indrel_1mes'] = 'Co-owner'
final_month.loc[final_month.indrel_1mes == '2.0', 'indrel_1mes'] = 'Co-owner'
final_month.loc[final_month.indrel_1mes == 2, 'indrel_1mes'] = 'Co-owner'
final_month.loc[final_month.indrel_1mes == 2.0, 'indrel_1mes'] = 'Co-owner'
final_month.loc[final_month.indrel_1mes == '3', 'indrel_1mes'] = 'Former Primary'
final_month.loc[final_month.indrel_1mes == '3.0', 'indrel_1mes'] = 'Former Primary'
final_month.loc[final_month.indrel_1mes == 3, 'indrel_1mes'] = 'Former Primary'
final_month.loc[final_month.indrel_1mes == 3.0, 'indrel_1mes'] = 'Former Primary'
final_month.loc[final_month.indrel_1mes == '4', 'indrel_1mes'] = 'Former Co-owner'
final_month.loc[final_month.indrel_1mes == '4.0', 'indrel_1mes'] = 'Former Co-owner'
final_month.loc[final_month.indrel_1mes == 4, 'indrel_1mes'] = 'Former Co-owner'
final_month.loc[final_month.indrel_1mes == 4.0, 'indrel_1mes'] = 'Former Co-owner'

final_month.indrel_1mes = final_month.indrel_1mes.fillna('Primary')

print("Step 7/13")
final_month.nomprov = final_month.nomprov.fillna("MADRID")

print("Step 8/13")
final_month.renta = final_month.renta.astype(float)
final_month.loc[final_month.renta.isnull(), "renta"] = final_month.groupby('nomprov').renta.transform('median')

print("Step 9/13")
final_month.segmento = final_month[final_month.renta <= 98000].segmento.fillna("03 - UNIVERSITARIO")
final_month.segmento = final_month[final_month.renta <= 125500].segmento.fillna("02 - PARTICULARES")
final_month.segmento = final_month.segmento.fillna("01 - TOP")

print("Step 10/13")
final_month.sexo = final_month.sexo.fillna("V")

print("Step 11/13")
final_month.tiprel_1mes = final_month.tiprel_1mes.fillna('I')

print("Step 12/13")
final_month.ind_nomina_ult1 = final_month.ind_nomina_ult1.fillna(0.0)
final_month.ind_nom_pens_ult1 = final_month.ind_nom_pens_ult1.fillna(0.0)

print("Step 13/13")
final_month.loc[final_month.antiguedad == -999999, 'antiguedad'] = final_month[final_month.antiguedad >= 0].antiguedad.median()

#Feature Engineering

print("Step 1/10")
final_month.age = final_month.age.astype(int)
final_month.loc[final_month.age < 20,"age"] = 19
final_month.loc[final_month.age > 90,"age"] = 91

print("Step 2/10")
final_month['ageGroup'] = (final_month.age // 10) - 1

print("Step 3/10")
final_month['isSpanish'] = final_month.pais_residencia.map(lambda x: 1 if x == "ES" else 0)

print("Step 4/10")
final_month['majorCity'] = final_month.nomprov.map(lambda x: 1 if x == "MADRID" or x == "BARCELONA" else 0)

print("Step 5/10")
final_month['fecha_alta_year'] = pd.DatetimeIndex(final_month.fecha_alta).year - 1995
final_month['fecha_dato_year'] = pd.DatetimeIndex(final_month.fecha_dato).year - 2015
final_month['fecha_alta_month'] = pd.DatetimeIndex(final_month.fecha_alta).month - 1
final_month['fecha_dato_month'] = pd.DatetimeIndex(final_month.fecha_dato).month - 1

print("Step 6/10")
final_month.antiguedad = final_month.antiguedad.astype(int)
final_month['antiguedad_years'] = final_month.antiguedad // 12

print("Step 7/10")
final_month.loc[final_month.indrel == 99.0, "indrel"] = 0.0

print("Step 8/10")
final_month['HAS_ult_fec_cli_1t'] = final_month.ult_fec_cli_1t.map(lambda x: 1 if(x) else 0)

print("Step 9/10")
final_month = final_month.drop('ult_fec_cli_1t', 1)

print("Step 10/10")
final_month['rentaGroup'] = final_month.renta.astype(float) // 50000
final_month.loc[final_month.renta >= 1000000, "rentaGroup"] = 20
final_month.loc[final_month.renta >= 10000000, "rentaGroup"] = 21


final_month_training_cols = final_month
final_month_training_cols = final_month_training_cols.drop(target_cols_all, axis=1)


print("Step 1/6")
final_month.pais_residencia = final_month.pais_residencia.astype('category').cat.codes
print("Step 2/6")
final_month.canal_entrada = final_month.canal_entrada.astype('category').cat.codes
print("Step 3/6")
final_month.nomprov = final_month.nomprov.astype('category').cat.codes
print("Step 4/6")
final_month.nomprov = final_month.indrel_1mes.astype('category').cat.codes
print("Step 5/6")
final_month = pd.get_dummies(final_month, columns = ['ind_empleado','sexo','tiprel_1mes','indresi',
                                   'indext','conyuemp','indfall','segmento','indrel_1mes'])
print("Step 6/6")
final_month = final_month.drop(['fecha_dato', 'fecha_alta'], axis = 1)



print("Step 1/11")
#Get the target columns
labels_final_month = final_month[target_cols_all]

print("Step 2/11")
#Add ncodpers to the dataframe
labels_final_month['ncodpers'] = final_month.ncodpers

print("Step 3/11")
labels_final_month = labels_final_month.set_index("ncodpers")

print("Step 4/11")
stacked_labels_final_month = labels_final_month.stack()

print("Step 5/11")
filtered_labels_final_month = stacked_labels_final_month.reset_index()

print("Step 6/11")
filtered_labels_final_month.columns = ["ncodpers", "product", "newly_added"]

print("Step 7/11")
#Only select the rows where there is a new product.
filtered_labels_final_month = filtered_labels_final_month[filtered_labels_final_month["newly_added"] == 1.0]

print("Step 8/11")
#Merge with the training features.
multiclass_final_month = filtered_labels_final_month.merge(final_month_training_cols, on="ncodpers", how="left")

print("Step 9/11")
final_month = multiclass_final_month

print("Step 10/11")
labels_final_month_final = final_month['product']

print("Step 11/11")
final_month_ncodpers = final_month.ncodpers
#Remove the columns that are not needed to train the model.
final_month = final_month.drop(['ncodpers','newly_added','product'], axis = 1)


# In[28]:


print(len(train_final))
print(len(labels_final))
print(len(final_month))
print(len(labels_final_month_final))


# In[29]:


labels_final.value_counts()


# ## Build a Model

# In[30]:


labels_final = labels_final.astype('category').cat.codes


# In[31]:


print(len(train_final.columns))
print(len(test.columns))


# In[32]:


print(train_final.columns)
print
print(test.columns)


# In[33]:


test = test.drop('ncodpers', axis = 1)


# In[34]:


train_final = train_final.drop(['tipodom'],axis = 1)
train_final = train_final.drop(['ind_empleado_S'],axis = 1)
train_final = train_final.drop(['indresi_N'], axis = 1)
train_final = train_final.drop(['indresi_S'], axis = 1)
train_final = train_final.drop(['conyuemp_S'], axis = 1)
train_final = train_final.drop(['conyuemp_Unknown'], axis = 1)
train_final = train_final.drop(['indfall_S'], axis = 1)
train_final = train_final.drop(['indrel_1mes_Co-owner'], axis = 1)
train_final = train_final.drop(['indrel_1mes_Former Primary'], axis = 1)
train_final = train_final.drop(['indrel_1mes_Primary'], axis =1)


# In[35]:


test = test.drop(['tipodom'], axis = 1)
test = test.drop(['ind_empleado_S'], axis = 1)
test = test.drop(['indresi_N'], axis = 1)
test = test.drop(['indresi_S'], axis = 1)
test = test.drop(['conyuemp_S'], axis = 1)
test = test.drop(['conyuemp_Unknown'], axis = 1)
test = test.drop(['indfall_S'], axis = 1)
test = test.drop(['indrel_1mes_Co-owner'], axis = 1)
test = test.drop(['indrel_1mes_Former Primary'], axis = 1)
test = test.drop(['indrel_1mes_Primary'], axis = 1)


# In[36]:


test.head()


# In[37]:


import warnings
warnings.filterwarnings("ignore")

xgtrain = xgb.DMatrix(train_final, label = labels_final)
xgtest = xgb.DMatrix(test, label = labels_final)
watchlist = [(xgtrain, 'train')]


# In[38]:


random_state = 4
params = {
        'eta': 0.05,
        'max_depth': 6,
        'min_child_weight': 4,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'gamma': 0,
        'alpha': 0,
        'lambda': 1,
        'verbose_eval': True,
        'seed': random_state,
        'num_class': 24,
        'objective': "multi:softprob",
        'eval_metric': 'mlogloss'
    }


# In[39]:


iterations = 40
printN = 1
#early_stopping_rounds = 10

xgbModel = xgb.train(params, 
                      xgtrain, 
                      iterations, 
                      watchlist,
                      verbose_eval = printN
                      #early_stopping_rounds=early_stopping_rounds
                      )


# In[40]:


import operator
importance = xgbModel.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))
print(importance)
print(len(importance))


# In[41]:


labels_final_month_final_cat = labels_final_month_final.astype('category').cat.codes


# In[42]:


used_products = pd.DataFrame()
used_products['product'] = labels_final_month_final_cat
used_products['ncodpers'] = final_month_ncodpers
used_products = used_products.drop_duplicates(keep = 'last')


# In[43]:


#create a dictionary to store each product a customer already has
used_recommendation_products = {}
target_cols_all = np.array(target_cols_all)
#iterate through used_products and add each one to used_recommendation_products
for idx,row_val in used_products.iterrows():
    used_recommendation_products.setdefault(row_val['ncodpers'],[]).append(target_cols_all[row_val['product']])
    if len(used_recommendation_products) % 100000 == 0:
        print(len(used_recommendation_products))


# In[44]:


len(used_recommendation_products)


# In[45]:


used_recommendation_products[15889]


# In[46]:


XGBpreds = xgbModel.predict(xgtest)


# In[47]:


XGBpreds


# In[48]:


pred = np.argsort(XGBpreds, axis=1)
pred = np.fliplr(pred)


# In[49]:


pred[0]


# In[50]:


#test_ids are the customer codes for the testing data.
test_ids = np.array(pd.read_csv("gs://chatrath/test_ver2.csv",usecols=['ncodpers'])['ncodpers'])
target_cols_all = np.array(target_cols_all)
final_preds = []
#iterate through our model's predictions (pred) and add the 7 most recommended products that the customer does not have.
for idx,predicted in enumerate(pred):
    ids = test_ids[idx]
    top_product = target_cols_all[predicted]
    used_products = used_recommendation_products.get(ids,[])
    new_top_product = []
    for product in top_product:
        if product not in used_products:
            new_top_product.append(product)
        if len(new_top_product) == 7:
            break
    final_preds.append(' '.join(new_top_product))
    if len(final_preds) % 100000 == 0:
        print(len(final_preds))


# In[51]:


final_preds


# In[53]:


submission = pd.DataFrame({'ncodpers':test_ids,'added_products':final_preds})
submission.to_csv('submission.csv',index=False)


# In[54]:


get_ipython().system("gsutil cp 'submission.csv' 'gs://chatrath/submission.csv'")


# In[ ]:




