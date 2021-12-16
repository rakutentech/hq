import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
loss functions used in the project
'''



def quantization_loss( C, output_img, code_img, output_txt, code_txt):
    '''
    quantization_loss quantization loss in paper equation(6)
    args:
        C: uantization centers in the dictionary
        output_img: image continuous feature
        code_img: image binary code
        output_txt: text continuous feature
        code_txt: text binary code
    '''
    img_loss = torch.sum(torch.mul(output_img-torch.matmul(code_img,C),output_img-torch.matmul(code_img,C)))
    txt_loss = torch.sum(torch.mul(output_txt-torch.matmul(code_txt,C), output_txt-torch.matmul(code_txt,C)))
    q_loss = img_loss+txt_loss
    return q_loss

def h_loss(output_img, output_txt,device=torch.device("cuda")):
    '''
    h_loss hash loss in paper equation(4) and balance loss in paper equation(5)
    args:
        output_img: image continuous feature
        output_txt: text continuous feature
        device: torch environment
    '''
    t_ones = torch.ones(output_img.shape).to(device, dtype=torch.float)
    
    fro_n = torch.norm(output_img*t_ones+output_txt*t_ones,p=2,dim=1)
    balance_loss = torch.sum(fro_n)
    Bx = (output_img >= 0).to(device,dtype=torch.float)
    By = (output_txt >= 0).to(device,dtype=torch.float)

    hash_loss = nn.MSELoss()(Bx,output_img)+nn.MSELoss()(By,output_txt)
    
    return balance_loss, hash_loss

def similarity_loss(f1, f2, sim, alpha=1,device=torch.device("cuda")):
    '''
    similarity_loss similarity loss in paper equation(3) 
    args:
        f1: one modal/domain's continuous feature of size batch_size * f_dim
        f2: another modal/domain's continuous feature
        sim: similarity score between f1 and f2, can only be 0 or 1, of size batch_size * 1
        alpha: hyper-parameter in the equation
    '''
 
    batch_size, f_dim = f1.shape
    inner = torch.bmm(f1.view(batch_size, 1, f_dim), f2.view(batch_size, f_dim, 1))
    #inner = torch.clamp(torch.bmm(f1.view(batch_size, 1, f_dim), f2.view(batch_size, f_dim, 1)),-1.5e1, 1.5e1)
    t_ones = torch.ones(batch_size,1).to(device, dtype=torch.float)
    similarity_loss = torch.mean(torch.log(torch.add(t_ones,torch.exp(inner)))- alpha*torch.mul(sim, inner))
    return similarity_loss

