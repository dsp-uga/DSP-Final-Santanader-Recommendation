# necessary import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.linalg import sqrtm
from copy import deepcopy
import argparse

def loader(path):
    # column useful for prediction
    usecols1 = ['fecha_dato', 'ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
           'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
           'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
           'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
           'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
           'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
           'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
           'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
    
    path_train=path+'train_ver2.csv'
    path_test=path+'test_ver2.csv'
    print(path_train)
    print(path_test)
    #loading the data into the dataframe with selected column
    train = pd.read_csv(path_train, usecols=usecols1)
    # loading with particular data in the dataframe
    train1 = train[train['fecha_dato']=="2016-05-28"].drop("fecha_dato", axis = 1)
    train2 = train[train['fecha_dato']=="2016-04-28"].drop("fecha_dato", axis = 1)
    # creating the deep copy 
    true = deepcopy(train1)
    #reading test file
    test = pd.read_csv(path_test)
    test = test[['ncodpers']]
    print("datasets loaded")
    #creating the list of user
    users = true['ncodpers'].tolist()
    true.drop('ncodpers', axis=1, inplace=True)
    # creating dict of user with the product
    items = true.columns.tolist()
    u = {}
    for i in range(len(users)):
        u[users[i]] = i
    trueMat = np.array(true)
    print(u)
    print("users dict formed")
    return train1,train2,test,trueMat,users,u
    
def reorder(train,users):
    train.index = train['ncodpers'].tolist()
    train.drop('ncodpers', axis=1, inplace=True)
    train = train.reindex(users)
    return train

# computing SVD
def svd(train, trueMat, k):
    utilMat = np.array(train)

    mask = np.isnan(utilMat)
    masked_arr=np.ma.masked_array(utilMat, mask)
    item_means=np.mean(masked_arr, axis=0)
    utilMat = masked_arr.filled(item_means)
    x = np.tile(item_means, (utilMat.shape[0],1))
    utilMat = utilMat - x
    print(utilMat)
    U, s, V=np.linalg.svd(utilMat, full_matrices=False)
    s=np.diag(s)
    s=s[0:k,0:k]
    U=U[:,0:k]
    V=V[0:k,:]
    s_root=sqrtm(s)

    Usk=np.dot(U,s_root)
    skV=np.dot(s_root,V)
    UsV = np.dot(Usk, skV)

    UsV = UsV + x

    UsV = np.ma.masked_where(trueMat==1,UsV).filled(fill_value=-999)

    #print(UsV)
    print("svd done")
    return UsV

# taking out the maximum element
def max_items(UsV,x,j):
    out = []

    for xx in x:
        if UsV[j,xx]>0.001: # setting a threshold
            out.append(items[xx])

    return out

#recommendation system 
def recommend(test,trueMat,train1,train2,u):

    UsV = (5*svd(train=train1,trueMat=trueMat,k=4) + 5*svd(train=train2, trueMat=trueMat, k=4))/10
    
    pred = []
    testusers = test['ncodpers'].tolist()
    n = 7
    for user in testusers[:]:
        j = u[user]
        p = max_items(UsV, UsV[j,:].argsort()[-n:][::-1], j)
        pred.append(" ".join(p))
        print(len(p))
    test['added_products'] = pred
    test.to_csv('sub.csv', index=False)
    
#main
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description=('Trains the model and outputs predictions.'),
        add_help='How to use', prog='svd.py <args>')
    
    # Required arguments
    parser.add_argument("--data_path", required=True,
                        help=("Provide the path to the data folder"))
    
    #parser.add_argument('--model', type=str, choices=['random_forest', 'logistic_regression', 'navies_bayes', 'svm','kmean'], 
    #default='random_forest', help = 'model to use for spam classification')
    
    args = vars(parser.parse_args())
    
    # Getting the names of the training / testing files.
    path=args['data_path']
    train1,train2,test,trueMat,users,u=loader(path)
    train2 = reorder(train2,users)
    train1 = reorder(train1,users)
    recommend(test,trueMat,train1,train2,u)

