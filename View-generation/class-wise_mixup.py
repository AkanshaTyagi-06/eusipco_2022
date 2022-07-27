#importing libraries
import numpy as np

fea_path = "..../l3net/features/"
path = "..../l3net/views_data/"


l_v = ['b','h','l','p','s','v']

for i in range(6):
    #path to read files
    txt_path = open(path+l_v[i]+"/class-wise_txt/airport(class name).txt") #path to txt file of a particular class
    
    #path to save final class file (mixup+original)
    save_npy_path = path+l_v[i]+"/class-wise_npy/airport"
    
    txt_files = txt_path.readlines()
    
    
    
    original_len = len(txt_files) # actual samples
    desired_len  = 132            # max samples
    mixup_number = desired_len - original_len # samples to be added
    print("mixup number",mixup_number)
    
    
    mat_final = np.zeros((desired_len,512))
    mat_index = 0
    
    
    ''' storing actual samples into final matrix'''
    for i in range(original_len):
        temp = txt_files[i].split(".")[0]  
        fe = np.load(fea_path+temp+".npy").mean(axis=0)
        mat_final[mat_index,:] = fe
        mat_index+= 1
    
    
    ''' mix-up preparation '''
    np.random.seed(0)
    ri_1 = np.random.randint(0,original_len,size = mixup_number) # random indexes
    ri_2 = np.random.randint(0,original_len,size = mixup_number) # random indexes
    mix_coef = np.random.uniform(low=0.0, high=1.0, size = mixup_number) # mix-up coefficients
    
    
    ''' mixing-up''' 
    for j in range(mixup_number):
        t1 = mat_final[ri_1[j]]    
        t2 = mat_final[ri_2[j]]
        alpha = mix_coef[j]
        mix_fea = (1-alpha) * t1 + (alpha * t2)   
        mat_final[mat_index,:] = mix_fea
        mat_index+= 1
    
    #saving the final matrix
    np.save(save_npy_path,mat_final)
    
    print("mat final",mat_final[131])

