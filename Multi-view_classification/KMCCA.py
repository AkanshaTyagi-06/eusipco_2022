from mvlearn.embed import KMCCA
import scipy.io
import numpy as np

# creating list of dictionaries
path = "path to views data"
b = scipy.io.loadmat(path+'view_b.mat')
h= scipy.io.loadmat(path+'view_h.mat')
l =scipy.io.loadmat(path+'view_l.mat')
p = scipy.io.loadmat(path+'view_p.mat')
s = scipy.io.loadmat(path+'view_s.mat')
v = scipy.io.loadmat(path+'view_v.mat')

views_list = [b,h,l,p,s,v] # list of dictionaries of all the views

train_data = [] # contains train data for all the views
test_data = []  # contains test data for all the views
train_label = [] # contains train label for all the views
test_label = [] # contains test label for all the views

for i in range(len(views_list)):
    print("i",i)
    train_data.append(views_list[i]['train_data'])
    test_data.append(views_list[i]['test_data'])
    train_label.append(views_list[i]['train_label'])
    test_label.append(views_list[i]['test_label'])
 
n_com = 9 # dimension of multiview feature
kmcca = KMCCA(n_components=n_com,kernel='linear',regs=0.95,tol=0.1)
kmcca.fit(train_data)
print("eigen values",kmcca.evals_)
#computing transformation of train data of every view w.r.t every view
transform_train = []
for i in range(len(views_list)):
    print("i",i)
    mat_cols = n_com * len(views_list)
    mat_rows = train_data[i].shape[0]
    print("rows, cols",mat_rows,mat_cols)
    mat = np.zeros((mat_rows,mat_cols))
    s = 0
    e = n_com   
    for j in range(len(views_list)):
        print("s e",s,e)
        mat[:,s:e] = kmcca.transform_view(train_data[i],j)
        s = e
        e = e + n_com
    transform_train.append(mat)
#print(len(transform_train))
#print(transform_train[5].shape)
   
#computing transformation of test data of every view w.r.t every view
transform_test = []
for i in range(len(views_list)):
    print("i",i)
    mat_cols = n_com * len(views_list)
    mat_rows = test_data[i].shape[0]
    mat = np.zeros((mat_rows,mat_cols))
    s = 0
    e = n_com
    for j in range(len(views_list)):
        print("s e",s,e)
        mat[:,s:e] = kmcca.transform_view(test_data[i],j)
        s = e
        e = e + n_com
    transform_test.append(mat)
#print(len(transform_test))
#print(transform_test[1].shape)
final_train = np.concatenate((np.asarray(transform_train[0]),np.asarray(transform_train[1]),\
                              np.asarray(transform_train[2]),np.asarray(transform_train[3]),\
                              np.asarray(transform_train[4]),np.asarray(transform_train[5])),axis=0)
#print(final_train.shape)  
final_test = np.concatenate((np.asarray(transform_test[0]),np.asarray(transform_test[1]),\
                              np.asarray(transform_test[2]),np.asarray(transform_test[3]),\
                              np.asarray(transform_test[4]),np.asarray(transform_test[5])),axis=0)
#print(final_test.shape)       
label_train =  np.concatenate((np.asarray(train_label[0].flatten()),np.asarray(train_label[1].flatten()),\
                              np.asarray(train_label[2].flatten()),np.asarray(train_label[3].flatten()),\
                              np.asarray(train_label[4].flatten()),np.asarray(train_label[5].flatten())),axis=0) 
#print("label_train",label_train.shape)     
label_test =  np.concatenate((np.asarray(test_label[0].flatten()),np.asarray(test_label[1].flatten()),\
                              np.asarray(test_label[2].flatten()),np.asarray(test_label[3].flatten()),\
                              np.asarray(test_label[4].flatten()),np.asarray(test_label[5].flatten())),axis=0) 
#print("label_train",label_test.shape) 
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(final_train, label_train)
pred = neigh.predict(final_test)

from sklearn.metrics import accuracy_score
print("acc multiview",accuracy_score(label_test, pred))

original_train = np.concatenate((train_data[0],train_data[1],train_data[2],\
                                 train_data[3],train_data[4],train_data[5]),axis=0)

original_test = np.concatenate((test_data[0],test_data[1],test_data[2],\
                                 test_data[3],test_data[4],test_data[5]),axis=0)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(original_train, label_train)
pred = neigh.predict(original_test)
print("acc non-multiview",accuracy_score(label_test, pred))


        
        

    
