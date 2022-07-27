path = "path where views data is stored"

l_v = ['b','h','l','p','s','v']

for i in range(6):
    print("i",i,path+l_v[i]+"/txt/"+l_v[i]+"_train.txt")
    f= open(path+l_v[i]+"/txt/"+l_v[i]+"_train.txt")
    data =f.readlines()
    print("data length",len(data))
    
    #creating list for every class
    l_a=[]
    l_m=[]
    l_ms=[]
    l_b=[]
    l_p=[]
    l_ps=[]
    l_sm=[]
    l_sp=[]
    l_t=[]
    l_st=[]
        
    #file instances for every class
    f_a=open(path+l_v[i]+"/class-wise_txt/airport.txt","w")
    f_b=open(path+l_v[i]+"/class-wise_txt/bus.txt","w")
    f_m=open(path+l_v[i]+"/class-wise_txt/metro.txt","w")
    f_ms=open(path+l_v[i]+"/class-wise_txt/metro_station.txt","w")
    f_p=open(path+l_v[i]+"/class-wise_txt/park.txt","w")
    f_ps=open(path+l_v[i]+"/class-wise_txt/public_square.txt","w")  
    f_sm=open(path+l_v[i]+"/class-wise_txt/shopping_mall.txt","w")
    f_sp=open(path+l_v[i]+"/class-wise_txt/street_pedestrian.txt","w")
    f_st=open(path+l_v[i]+"/class-wise_txt/street_traffic.txt","w")
    f_t=open(path+l_v[i]+"/class-wise_txt/tram.txt","w")
  
    #adding data in city-wise list
    for i in range(len(data)):
        t = data[i].split("-")
        label = t[0].split("/")[1]
        #print("label",label)
        city = t[1]
        #print("city",city)
        
        if label =="airport":
            l_a.append(data[i])
            
        if label =="metro":
            l_m.append(data[i])
            
        if label =="metro_station":
            l_ms.append(data[i])
            
        if label =="bus":
            l_b.append(data[i])
            
        if label =="park":
            l_p.append(data[i])
            
        if label =="public_square":
            l_ps.append(data[i])
            
        if label =="shopping_mall":
            l_sm.append(data[i])
            
        if label =="street_pedestrian":
            l_sp.append(data[i])
            
        if label =="tram":
            l_t.append(data[i])
            
        if label =="street_traffic":
            l_st.append(data[i])
            
    #writing data in txt files
    for e in l_a:
        f_a.write(e)
        
    for e in l_m:
        f_m.write(e)
        
    for e in l_ms:
        f_ms.write(e)
        
    for e in l_b:
        f_b.write(e)
        
    for e in l_p:
        f_p.write(e)
        
    for e in l_ps:
        f_ps.write(e)
        
    for e in l_sm:
        f_sm.write(e)
        
    for e in l_sp:
        f_sp.write(e)
        
    for e in l_t:
        f_t.write(e)
        
    for e in l_st:
        f_st.write(e)
        

         
    
