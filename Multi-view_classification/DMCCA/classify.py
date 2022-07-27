"""
Created on Sat Jan  1 17:18:24 2022

@author: akansha
"""
from sklearn import preprocessing
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import scipy.io
import numpy as np


#for non-multiview features
path = 'path to views data'
barcelona = scipy.io.loadmat(path+'view_b.mat')
helsinki = scipy.io.loadmat(path+'view_h.mat')
london = scipy.io.loadmat(path+'view_l.mat')
paris = scipy.io.loadmat(path+'view_p.mat')
stockholm = scipy.io.loadmat(path+'view_s.mat')
vienna = scipy.io.loadmat(path+'view_v.mat')

#collecting labels
label_tr_b = barcelona['train_label'].squeeze()
label_te_b = barcelona['test_label'].squeeze()

label_tr_h = helsinki['train_label'].squeeze()
label_te_h = helsinki['test_label'].squeeze()

label_tr_l = london['train_label'].squeeze()
label_te_l = london['test_label'].squeeze()

label_tr_p = paris['train_label'].squeeze()
label_te_p = paris['test_label'].squeeze()

label_tr_s = stockholm['train_label'].squeeze()
label_te_s = stockholm['test_label'].squeeze()

label_tr_v = vienna['train_label'].squeeze()
label_te_v = vienna['test_label'].squeeze()

#collecting data
data_tr_b = barcelona['train_data'].squeeze()
data_te_b = barcelona['test_data'].squeeze()

data_tr_h = helsinki['train_data'].squeeze()
data_te_h = helsinki['test_data'].squeeze()

data_tr_l = london['train_data'].squeeze()
data_te_l = london['test_data'].squeeze()

data_tr_p = paris['train_data'].squeeze()
data_te_p = paris['test_data'].squeeze()

data_tr_s = stockholm['train_data'].squeeze()
data_te_s = stockholm['test_data'].squeeze()

data_tr_v = vienna['train_data'].squeeze()
data_te_v = vienna['test_data'].squeeze()

nmf_train_data= np.concatenate((data_tr_b,data_tr_h,data_tr_l,data_tr_p,data_tr_s,data_tr_v),axis=0)
train_label = np.concatenate((label_tr_b,label_tr_h,label_tr_l,label_tr_p,label_tr_s,label_tr_v),axis=0).flatten()

nmf_test_data = np.concatenate((data_te_b,data_te_h,data_te_l,data_te_p,data_te_s,data_te_v),axis=0)
test_label = np.concatenate((label_te_b,label_te_h,label_te_l,label_te_p,label_te_s,label_te_v),axis=0).flatten()

memb_path = "path to DMCCA features/"

''' Multiview features for train data'''
train_v1 = np.load(memb_path+"emb_train_v1.npy")
#train_v1 = train_v1[:,v1_s:v1_e]
print(train_v1.shape)
train_v2 = np.load(memb_path+"emb_train_v2.npy")
#train_v2 = train_v2[:,v2_s:v2_e]
print(train_v2.shape)
train_v3 = np.load(memb_path+"emb_train_v3.npy")
#train_v3 = train_v3[:,v3_s:v3_e]
print(train_v3.shape)
train_v4 = np.load(memb_path+"emb_train_v4.npy")
#train_v1 = train_v1[:,v1_s:v1_e]
print(train_v4.shape)
train_v5 = np.load(memb_path+"emb_train_v5.npy")
#train_v2 = train_v2[:,v2_s:v2_e]
print(train_v5.shape)
train_v6 = np.load(memb_path+"emb_train_v6.npy")
#train_v3 = train_v3[:,v3_s:v3_e]
print(train_v6.shape)

''' Multiview features for test data'''
test_v1 = np.load(memb_path+"emb_test_v1.npy")
#test_v1 = test_v1[:,v1_s:v1_e]
#print(test_v1.shape)
test_v2 = np.load(memb_path+"emb_test_v2.npy")
#test_v2 = test_v2[:,v2_s:v2_e]
#print(test_v2.shape)
test_v3 = np.load(memb_path+"emb_test_v3.npy")
#test_v3 = test_v3[:,v3_s:v3_e]
#print(test_v3.shape)
test_v4 = np.load(memb_path+"emb_test_v4.npy")
#test_v1 = test_v1[:,v1_s:v1_e]
#print(test_v1.shape)
test_v5 = np.load(memb_path+"emb_test_v5.npy")
#test_v2 = test_v2[:,v2_s:v2_e]
#print(test_v2.shape)
test_v6 = np.load(memb_path+"emb_test_v6.npy")
#test_v3 = test_v3[:,v3_s:v3_e]
print(test_v3.shape)
 
mf_train_data= np.concatenate((train_v1,train_v2,train_v3,train_v4,train_v5,train_v6),axis=0)
mf_test_data = np.concatenate((test_v1,test_v2,test_v3,test_v4,test_v5,test_v6),axis=0)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

neigh_nmf = KNeighborsClassifier(n_neighbors=5)
neigh_nmf.fit(nmf_train_data, train_label)
pred_nmf = neigh_nmf.predict(nmf_test_data)
print("acc non-multiview",accuracy_score(test_label, pred_nmf))


neigh_mf = KNeighborsClassifier(n_neighbors=5)
neigh_mf.fit(mf_train_data, train_label)
pred_mf = neigh_mf.predict(mf_test_data)
print("acc multiview",accuracy_score(test_label, pred_mf))










