import numpy as np
import random
from sklearn.cluster import MiniBatchKMeans


'''
quantization functions 
X = sum_i^MC_ib_i  
X is continuous feature(input) 
C is center(output) 
b is binary code(output) 
'''

def initial_centers(img_input, txt_input,M,K,q_dim):
    '''
    initial_centers initial the quantization centers

    args:
        img_input: image continuous feature
        txt_input: text continuous feature
        M: number of dictionaries
        K: number of centers in each dictionary
        q_dim: continuous feature dimension


    '''
    C_init = np.zeros([M * K, q_dim])
    print("initilizing Centers")
    all_input = np.vstack([img_input.cpu().detach().numpy(), txt_input.cpu().detach().numpy()])
    
    for i in range(M):
        kmeans = MiniBatchKMeans(n_clusters=K).fit(all_input[:, int(i * q_dim / M) : int((i + 1) * q_dim / M)])
        C_init[i * K: (i + 1) * K, int(i * q_dim / M): int((i + 1) * q_dim / M)] = kmeans.cluster_centers_
        print("codebook: ", i, " finish")
    return C_init

def update_centers(img_input, img_code, txt_input, txt_code,M,K):
    '''
    update_centers update the centers given binary codes

    Optimize:
        self.C = (U * hu^T + V * hv^T) (hu * hu^T + hv * hv^T)^{-1}
        self.C^T = (hu * hu^T + hv * hv^T)^{-1} (hu * U^T + hv * V^T)
        but all the C need to be replace with C^T :
        self.C = (hu * hu^T + hv * hv^T)^{-1} (hu^T * U + hv^T * V)

    args:
        img_input: image continuous feature
        txt_input: text continuous feature
        img_code: image binary code
        txt_code: text binary code
        M: number of dictionaries
        K: number of centers in each dictionary

    '''
    img_code = img_code.cpu().numpy()
    txt_code = txt_code.cpu().numpy()
    img_input = img_input.cpu().detach().numpy()
    txt_input = txt_input.cpu().detach().numpy()
    
    h = np.concatenate((img_code, txt_code))
    U = np.concatenate((img_input, txt_input))
    smallResidual =np.eye(M*K, dtype = np.float32) * 0.001
    Uh = np.matmul(np.transpose(h), U)
    hh = np.add(np.matmul(np.transpose(h), h), smallResidual)
    compute_centers = np.matmul(np.linalg.inv(hh), Uh)
    
    C = compute_centers

    return C
def update_codes_ICM(output, code, C, max_iter_update_b,N,M,K):
        '''
        update_codes_ICM update the binary codes given the centers

        Optimize:
            min || output - self.C * codes ||
            min || output - codes * self.C ||
        args:
            output: [n_train, out_dim]
            C: [M*K, out_dim]
                [C_1, C_2, ... C_M]
            codes: [n_train, M*K]
        args:
        args:
            input: continuous feature
            code: binary code
            M: number of dictionaries
            K: number of centers in each dictionary
            N: number of input
            C: centers in a dictionary
            max_iter_update_b: max number of iterations for the updates
        '''
        code_np = code.cpu().numpy()
        output = output.cpu().detach().numpy()
        C = C.cpu().numpy()

        for iterate in range(max_iter_update_b):
            
            sub_list = [i for i in range(M)]
            random.shuffle(sub_list)
            for m in sub_list:
                # update the code in subspace m
                # dim: [subcenter * subspace, subcenter * subspace]
                v = np.zeros((N,K))
                for indicator in range(K):
                    a = np.zeros(K)
                    np.put(a,indicator,1)
                    code_np[:, m * K: (m + 1) * K] = a
                    v[:,indicator] = np.sum(np.square(output-np.matmul(code_np,C)),axis = 1)
                
                code_np[:, m * K: (m + 1) * K] = np.eye(K)[np.argmin(v,axis= 1).reshape(-1)]
                
                
        assert np.sum(np.sum(code_np, 1) == M), "update_code wrong"
            
        return code_np
