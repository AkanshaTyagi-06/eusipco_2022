import numpy as np
from scipy.io import savemat

l_v = ['b','h','l','p','s','v']
path = "path where the l3-net or soundnet features are stored"

for j in range(6):
    mat_save_path = "/l3net/views_data/"+l_v[j]+"/view_"+l_v[j]+".mat"
    tr_path = "/l3net/views_data/"+l_v[j]+"/class-wise_npy/"
    te_path = open("/l3net/views_data/"+l_v[j]+"/txt/"+l_v[j]+"_test.txt")
    
    '''for test data '''
    
    data_te = te_path.readlines()
    labels =["airport","metro","metro_station","bus","park","public_square","shopping_mall","street_pedestrian","tram","street_traffic"]
       
    test=[] # for data
    test_l=[] # for labels
    
    for i in range(len(data_te)):
        
        temp = data_te[i].split(".")[0]   
        fe = np.load(path+temp+".npy")    
        fe = fe.mean(axis=0)
        t = data_te[i].split("-")
        label = t[0].split("/")[1]
           
        if label == "airport":
            label = 0
        if label == "metro":
            label = 1
        if label == "metro_station":
            label = 2
        if label == "bus":
            label = 3
        if label == "park":
            label = 4
        if label == "public_square":
            label = 5
        if label == "shopping_mall":
            label = 6
        if label == "street_pedestrian":
            label = 7
        if label == "tram":
            label = 8
        if label == "street_traffic":
            label = 9
            
        test.append(fe)
        test_l.append(label)
        
    test = np.array(test)
    
    print(test.shape)
        
    '''for train data ''' 
    data_a = np.load(tr_path+"airport.npy")
    data_m = np.load(tr_path+"metro.npy")
    data_ms = np.load(tr_path+"metro_station.npy")
    data_b = np.load(tr_path+"bus.npy")
    data_p = np.load(tr_path+"park.npy")
    data_ps = np.load(tr_path+"public_square.npy")
    data_sm = np.load(tr_path+"shopping_mall.npy")
    data_sp = np.load(tr_path+"street_pedestrian.npy")
    data_t = np.load(tr_path+"tram.npy")
    data_st = np.load(tr_path+"street_traffic.npy")
    
    # concatenating all the class data together
    train = np.concatenate((data_a,data_m,data_ms,data_b,data_p,data_ps,data_sm,data_sp,data_t,data_st),axis=0)
    
    # generating train data labels
    train_l=[]
    
    for i in range(10):
        for j in range(132):
            train_l.append(i)
    
    print("train",train.shape)  
    print("test",test.shape)      
    print("train label",len(train_l))
    print("test label",len(test_l))
    
    
    # normalizing the data
    from sklearn.preprocessing import MinMaxScaler
     
    #print("before normalization",train[0])
    scaler = MinMaxScaler()
    scaler.fit(train)  
    train = scaler.fit_transform(train) 
    #print("after normalization",train[0])
    test = scaler.fit_transform(test)
    
    # saving the view data  
    mdic = {"train_data": train, "train_label": train_l,"test_data": test, "test_label": test_l}
    savemat(mat_save_path, mdic)


