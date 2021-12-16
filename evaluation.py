import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


'''
retrieval evaluation functions
'''


def check_sim(i_l, t_l):
    '''
    check_sim check if two entities are similar based on their labels
    args:
        i_l: image entitiy label
        t_l: text entitiy label
    '''
    sim = (torch.sum(i_l*t_l,dim=1)>0).to(dtype=torch.int8)
    return sim




def AQD_new(output,C, code, R,test_l, database_l,id_list):
    ''' 
    AQD_new: use quantization code for database, use dense feature for query to compute the distance,
             modified for HQ model
    args:
        output: dense feature of the query
        C: centers of the quantization dictionary
        code: quantization code of the database items
        R: mAP@R, consider precision results up to R-th top retrieval items
        test_l: test query's label
        database_l: databse item's label
        id_list: filtered ids from hashing comparion (step 1 in HQ)

    '''
    code = code.cpu().numpy()
    output = output.cpu().detach().numpy()
    C = C.cpu().numpy()
    
    
    APx = []
    print("#calc mAPs# calculating mAPs")
    for i in range(output.shape[0]):
        label = test_l[i, :]
        label[label == 0] = -1
        
        filtered_id = id_list[i]
        dis = np.dot(output[i], np.dot(code[filtered_id,:], C).T) 
        idx = np.argsort(-dis)
        imatch = np.sum(database_l[filtered_id[idx[0:R]], :] == label, 1) > 0
        rel = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)
        if rel != 0:
            APx.append(np.sum(Px * imatch) / rel)
    return np.mean(np.array(APx))



def simple(output,database, R,test_l, database_l):
    ''' 
    simple: use dense features for query and database data
    args:
        output: dense feature of the query
        database: dense feature of the database items
        R: mAP@R, consider precision results up to R-th top retrieval items
        test_l: test query's label
        database_l: databse item's label
    '''
    database = database.cpu().detach().numpy()
    output = output.cpu().detach().numpy()

    
    dis = np.dot(output, database.T)
    ids = np.argsort(-dis, 1)
    

    APx = []
    for i in range(output.shape[0]):
        label = test_l[i, :]
        label[label == 0] = -1
        idx = ids[i, :]
        
        imatch = np.sum(database_l[idx[0:R], :] == label, 1) > 0
        rel = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)
        if rel != 0:
            APx.append(np.sum(Px * imatch) / rel)
    return np.mean(np.array(APx))

def AQD(output,C, code, R,test_l, database_l):
    ''' 
    AQD: use quantization code for database, use dense feature for query to compute the distance,
         
    args:
        output: dense feature of the query
        C: centers of the quantization dictionary
        code: quantization code of the database items
        R: mAP@R, consider precision results up to R-th top retrieval items
        test_l: test query's label
        database_l: databse item's label
        id_list: filtered ids from hashing comparion (step 1 in HQ)

    '''
    code = code.cpu().numpy()
    output = output.cpu().detach().numpy()
    C = C.cpu().numpy()
    
    dis = np.dot(output, np.dot(code, C).T)
    ids = np.argsort(-dis, 1)
    

    APx = []
    for i in range(output.shape[0]):
        label = test_l[i, :]
        label[label == 0] = -1
        idx = ids[i, :]
        
        imatch = np.sum(database_l[idx[0:R], :] == label, 1) > 0
        rel = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)
        if rel != 0:
            APx.append(np.sum(Px * imatch) / rel)
    return np.mean(np.array(APx))



def HammingD(f1,  binary_f, R_h):
    ''' 
    HammingD: compare hamming distance for query and database
              modified for our model
    args:
        f1: dense feature of the query
        binary_f: binary hashing code of the database items
        R_h: only output top-R_h most similar items from the database
    '''
    f1 = f1.cpu().detach().numpy()
    binary_f = binary_f.cpu().numpy()
    A = 2*np.int8(f1>=0)-1
    B = 2*binary_f-1

    #print('b',B)
    dis = np.dot(A,np.transpose(B))
    ids = np.argsort(-dis, 1)
    APx = []
    ID = ids[:,0:R_h]

    return ID
    

    
### compare hamming distance for query and database   
def HammingD_only(f1,  binary_f, R, test_l, database_l):
    ''' 
    HammingD_only: compare hamming distance for query and database, 
                   compute mAP@R
    
                
    args:
        f1: dense feature of the query
        binary_f: binary hashing code of the database items
        R: mAP@R, consider precision results up to R-th top retrieval items
        test_l: test query's label
        database_l: databse item's label
    '''   
    f1 = f1.cpu().detach().numpy()
    binary_f = binary_f.cpu().numpy()
    A = 2*np.int8(f1>=0)-1
    B = 2*binary_f-1

    #print('b',B)
    dis = np.dot(A,np.transpose(B))
    ids = np.argsort(-dis, 1)
    APx = []

    
    for i in range(f1.shape[0]):
        label = test_l[i, :]
        label[label == 0] = -1
        idx = ids[i, :]
        
        imatch = np.sum(database_l[idx[0:R], :] == label, 1) > 0
        rel = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)
        if rel != 0:
            APx.append(np.sum(Px * imatch) / rel)
    
    return np.mean(np.array(APx))

