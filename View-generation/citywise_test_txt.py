# script to create test txt files city/view wise
f= open("path to test.txt") 
data =f.readlines()
print("data length",len(data))

#creating list for every city
l_b=[]
l_h=[]
l_l=[]
l_p=[]
l_s=[]
l_v=[]


#file instances for every city
file_path ="path to store view data"
f_b=open(file_path+"b/txt/b_test.txt","w")
f_h=open(file_path+"h/txt/h_test.txt","w")
f_l=open(file_path+"l/txt/l_test.txt","w")
f_p=open(file_path+"p/txt/p_test.txt","w")
f_s=open(file_path+"s/txt/s_test.txt","w")
f_v=open(file_path+"v/txt/v_test.txt","w")

#adding data in city-wise list
for i in range(len(data)):
    t = data[i].split("-")
    label = t[0].split("/")[1]
    #print("label",label)
    city = t[1]
    #print("city",city)
    loc = t[2]
    #print("location",loc)
    
    if city =="barcelona":
        l_b.append(data[i])
        
    if city =="helsinki":
        l_h.append(data[i])
        
    if city =="london":
        l_l.append(data[i])
            
    if city =="paris":
        l_p.append(data[i])
           
    if city =="stockholm":
        l_s.append(data[i])
        
    if city =="vienna":
        l_v.append(data[i])
        


#writing data in txt files
for e in l_b:
    f_b.write(e)
    
for e in l_h:
    f_h.write(e)
       
for e in l_l:
    f_l.write(e)
    
for e in l_p:
    f_p.write(e)
        
for e in l_s:
    f_s.write(e)
    
for e in l_v:
    f_v.write(e)
    

    
    
