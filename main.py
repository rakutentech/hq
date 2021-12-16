from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import random
import json



from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature

from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils import data
from PIL import Image

import sys
sys.path.append(".")
from load_data import load_dataset,MM_pre_feature_Dataset
from model import Net1, Net2
from losses import quantization_loss, h_loss, similarity_loss
from evaluation import check_sim, HammingD, AQD_new, simple
from utils import initial_centers, update_centers, update_codes_ICM

import argparse
########## Parameters ###############
parser = argparse.ArgumentParser(description='HQ')
parser.add_argument('--dataset', default='mirflickr', type=str, help='can be mirflickr or nuswide')
parser.add_argument('--eval_method', default='hq', type=str, help='can be hq or lossless')

parser.add_argument('--K', default=7, type=int, help='each dictionary size')
parser.add_argument('--M', default=1, type=int, help='number of dictionies')
parser.add_argument('--max_epochs', default=5, type=int)
parser.add_argument('--R_h', default=100, type=int, help='number of retrived items after hashing comparison')
parser.add_argument('--R', default=50, type=int, help='final MAP@R')
parser.add_argument('--log_interval', default=50, type=int)
parser.add_argument('--max_iter_update_Cb', default=1, type=int)
parser.add_argument('--max_iter_update_b', default=1, type=int)
parser.add_argument('--feature_dim', default=128, type=int, help='hidden feature dimension')

parser.add_argument('--img_lr', default=0.0005, type=float, help='initial image part lr')
parser.add_argument('--txt_lr', default=0.001, type=float, help='initial text part lr')
parser.add_argument('--q_lambda', default=0.0001, type=float, help='hyperparameter in the loss function')
parser.add_argument('--sim_lambda', default=30, type=float, help='hyperparameter in the loss function')
parser.add_argument('--h_lambda', default=0.6, type=float, help='hyperparameter in the loss function')
parser.add_argument('--b_lambda', default=0.0001, type=float, help='hyperparameter in the loss function')
args = parser.parse_args()



#### params is for data load ####
params = {'batch_size': 50,
          'shuffle': False,
          'num_workers': 0,
}

K =  2**args.K
M = args.M
q_dim = args.feature_dim
device = torch.device("cuda")

#### temparary parameters used in the training #######
max_val = [0,0]
max_epochs = args.max_epochs
log_interval = args.log_interval
find_better = 0
######### load the model #########   
if args.dataset == 'mirflickr':
    Net = Net2
if args.dataset == 'nuswide':
    Net = Net1
model = Net(args.feature_dim).to(device)
optimizer = optim.SGD([
                {'params': model.mlp.parameters()},
                {'params': model.alexnet.parameters(), 'lr': args.img_lr}
            ], lr=args.txt_lr, momentum=0.2)

######## load the dataset ###########
img, txt, labels = load_dataset(args)
train_image,vali_image,test_image  = img   
train_text,vali_text,test_text = txt
train_text_label,train_img_label, vali_label,test_labels = labels
images = np.concatenate((train_image,vali_image,test_image), axis=0)
texts =  np.concatenate((train_text,vali_text,test_text), axis=0)
text_labels = np.concatenate((train_text_label,vali_label,test_labels), axis=0)
image_labels = np.concatenate((train_img_label,vali_label,test_labels), axis=0)

train_n = train_img_label.shape[0]
test_n = test_labels.shape[0]
vali_n = vali_label.shape[0]
####### initiate parameters used in the training ######
C = torch.FloatTensor(M * K, q_dim).uniform_(-1, 1).to(device) 
# quantization codes
code_img_q = torch.zeros(train_n,M*K).to(device)
code_txt_q = torch.zeros(train_n,M*K).to(device)
# binary hashing features
code_img_h = torch.zeros((train_n, q_dim)).to(device)
code_txt_h = torch.zeros((train_n, q_dim)).to(device)
# dense features
img_output = torch.zeros(train_n, q_dim).to(device)
txt_output = torch.zeros(train_n,q_dim).to(device)

# retrival over all items
N_all = test_n + vali_n + train_n
img_output_all = torch.zeros(N_all, q_dim).to(device)
txt_output_all = torch.zeros(N_all,q_dim).to(device)
code_img_all_q = torch.zeros(N_all,M*K).to(device)
code_txt_all_q = torch.zeros(N_all,M*K).to(device)
code_img_all_h = torch.zeros(N_all,q_dim).to(device)
code_txt_all_h = torch.zeros(N_all,q_dim).to(device)


########## count the size of the model###########
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
count_parameters(model)




######### data Generators ##############
training_set = MM_pre_feature_Dataset(train_image, train_text, train_img_label,train_text_label)
training_generator = data.DataLoader(training_set, **params)
test_set = MM_pre_feature_Dataset(test_image, test_text, test_labels,test_labels)
test_generator = data.DataLoader(test_set, **params)

vali_set = MM_pre_feature_Dataset(vali_image, vali_text, vali_label,vali_label)
vali_generator = data.DataLoader(vali_set, **params)
##### check the whole dataset when retrival ###
all_set = MM_pre_feature_Dataset(images, texts, image_labels, text_labels)
all_generator = data.DataLoader(all_set, **params)




########## Start training #########
for epoch in range(max_epochs):
    avg_cost = 0
    avg_q_cost = 0
    avg_h_cost = 0
    avg_b_cost = 0
    avg_sim_cost = 0
    # Training
    model.train()
    for batch_idx, (local_batch, ) in enumerate(training_generator):
        # Transfer to GPU
        image,text = local_batch['image'].to(device,dtype=torch.float),local_batch['text'].to(device,dtype=torch.float)
        img_l = local_batch['img_l']
        txt_l = local_batch['txt_l']
        
        # Model computations
        optimizer.zero_grad()
        img_q, text_q = model(image,text)
        img_output[batch_idx*params['batch_size']: (batch_idx+1)*params['batch_size']] = img_q
        txt_output[batch_idx*params['batch_size']: (batch_idx+1)*params['batch_size']] = text_q
        
        code_img_h[batch_idx*params['batch_size']: (batch_idx+1)*params['batch_size']] = (img_q>=0).to(dtype=torch.int8)
        code_txt_h[batch_idx*params['batch_size']: (batch_idx+1)*params['batch_size']] = (text_q>=0).to(dtype=torch.int8)
               
        #compute loss
        q_loss = quantization_loss(C, img_q, code_img_q[batch_idx*params['batch_size']: (batch_idx+1)*params['batch_size']], text_q, code_txt_q[batch_idx*params['batch_size']: (batch_idx+1)*params['batch_size']]) 
        balance_loss, hash_loss = h_loss(img_q,text_q) 
        sim = check_sim(img_l,txt_l).to(device,dtype=torch.float)
        alpha = 1
        sim_loss = similarity_loss(img_q, text_q, sim, alpha)
        
        loss = args.q_lambda*q_loss + args.h_lambda*hash_loss+args.b_lambda*balance_loss+args.sim_lambda*sim_loss
        loss.backward()
        optimizer.step()
                    
        assert not np.isnan(sim_loss.item()), 'Model diverged with sim loss = NaN'
        assert not np.isnan(hash_loss.item()), 'Model diverged with hash loss = NaN'
        assert not np.isnan(balance_loss.item()), 'Model diverged with balance loss = NaN'
        assert not np.isnan(q_loss.item()), 'Model diverged with quantization loss = NaN'
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tquan Loss: {:.2f} \thash Loss: {:.2f}\t balance Loss: {:.2f}\tSimilarity Loss: {:.2f}'.format(
                epoch, batch_idx * len(image), len(training_generator.dataset), args.q_lambda*q_loss, args.h_lambda*hash_loss, args.b_lambda*balance_loss, args.sim_lambda*sim_loss))
            
        avg_q_cost += (q_loss) / train_n * params['batch_size']     
        avg_h_cost += (hash_loss) / train_n * params['batch_size'] 
        avg_b_cost += (balance_loss) / train_n * params['batch_size'] 
        avg_sim_cost += (sim_loss) / train_n * params['batch_size'] 
        avg_cost += (loss) / train_n * params['batch_size']   
    
    
    print('Train Epoch: {} \tAvg cost: {:.6f}'.format(epoch, avg_cost))  

    ### update quantization ######    
    if epoch > 0:
        if epoch == 1:
            #initiate dictionary
            C = torch.from_numpy(initial_centers(img_output, txt_output, M, K, q_dim) ).to(device)
        
        for i in range(args.max_iter_update_Cb):

            code_img_q = torch.from_numpy(update_codes_ICM(img_output, code_img_q, C, args.max_iter_update_b, train_n,M,K) ).to(device)
            code_txt_q = torch.from_numpy(update_codes_ICM(txt_output, code_txt_q, C, args.max_iter_update_b, train_n,M,K) ).to(device)               
            # update Centers
            C = torch.from_numpy(update_centers(img_output, code_img_q, txt_output, code_txt_q, M, K) ).to(device)
                        
        print("update centers and codes done!!!")
    
    ### evaluation ####     
    if epoch %10 == 0:            
        print('validation')
        model.eval()
        img_output_vali = torch.zeros(vali_n, q_dim).to(device)
        txt_output_vali = torch.zeros(vali_n,q_dim).to(device)
        with torch.set_grad_enabled(False):
            for batch_idx, (local_batch, ) in enumerate(vali_generator):
                # Transfer to GPU           
                image,text = local_batch['image'].to(device,dtype=torch.float),local_batch['text'].to(device,dtype=torch.float)
                img_q,text_q = model(image,text)
                img_output_vali[batch_idx*params['batch_size']: (batch_idx+1)*params['batch_size']] = img_q
                txt_output_vali[batch_idx*params['batch_size']: (batch_idx+1)*params['batch_size']] = text_q
            
            R_h = args.R_h
            R = 50
            if args.eval_method == 'hq':
                l = np.copy(vali_label)
                #image2text
                idx1 = HammingD(img_output_vali,  code_txt_h, R_h)
                #text2image
                idx2 = HammingD(txt_output_vali, code_img_h, R_h)
                mApx1 = AQD_new(img_output_vali, C, code_txt_q, R,l, train_text_label,idx1)
                mApx2 = AQD_new(txt_output_vali, C, code_img_q, R,l, train_img_label,idx2)

            
            if args.eval_method == 'lossless':
                l=np.copy(vali_label)
                mApx1 = simple(img_output_vali, txt_output, R,l, train_text_label)
                l=np.copy(vali_label)
                mApx2 = simple(txt_output_vali, img_output, R,l, train_img_label)
            
            
            print("vali image2text50"+ args.eval_method ,mApx1)
            print("vali text2image50"+ args.eval_method ,mApx2)

            #### only use this for validation sets #######
            if mApx1 > max_val[0]:
                find_better = 1
                max_val[0] = mApx1
                
                
            elif mApx2 > max_val[1]:
                find_better = 1
                max_val[1] = mApx2
        
        ######### if find best validation then we run test ##########
        if find_better == 1:
            print('all')
            find_better = 0
            model.eval()
            img_output_test = torch.zeros(test_n, q_dim).to(device)
            txt_output_test = torch.zeros(test_n,q_dim).to(device)
            R_h = args.R_h
            R = args.R
            with torch.set_grad_enabled(False):
                for batch_idx, (local_batch, ) in enumerate(all_generator):
                    # Transfer to GPU           
                    image,text = local_batch['image'].to(device,dtype=torch.float),local_batch['text'].to(device,dtype=torch.float)
                    #local_labels = local_labels.to(device, dtype=torch.float)
                    img_q,text_q = model(image,text)
                    img_output_all[batch_idx*params['batch_size']: (batch_idx+1)*params['batch_size']] = img_q
                    txt_output_all[batch_idx*params['batch_size']: (batch_idx+1)*params['batch_size']] = text_q

                    code_img_all_h[batch_idx*params['batch_size']: (batch_idx+1)*params['batch_size']] = (img_q>=0).to(dtype=torch.int8)
                    code_txt_all_h[batch_idx*params['batch_size']: (batch_idx+1)*params['batch_size']] = (text_q>=0).to(dtype=torch.int8)


                for batch_idx, (local_batch, ) in enumerate(test_generator):
                    # Transfer to GPU           
                    image,text = local_batch['image'].to(device,dtype=torch.float),local_batch['text'].to(device,dtype=torch.float)
                    #local_labels = local_labels.to(device, dtype=torch.float)

                    img_q,text_q = model(image,text)
                    img_output_test[batch_idx*params['batch_size']: (batch_idx+1)*params['batch_size']] = img_q
                    txt_output_test[batch_idx*params['batch_size']: (batch_idx+1)*params['batch_size']] = text_q

                l = np.copy(test_labels)
                labelsi = np.copy(image_labels)
                labelst = np.copy(text_labels)
                code_img_all_q = torch.from_numpy(update_codes_ICM(img_output_all, code_img_all_q, C, args.max_iter_update_b, N_all,M,K) ).to(device)
                code_txt_all_q = torch.from_numpy(update_codes_ICM(txt_output_all, code_txt_all_q, C, args.max_iter_update_b, N_all,M,K) ).to(device)
                

                if args.eval_method == 'hq':
                    #image2text
                    t1 = time.time()
                    idx1_all = HammingD(img_output_test,  code_txt_all_h, R_h)
                    mApx1 = AQD_new(img_output_test, C, code_txt_all_q, R,l, labelst,idx1_all)
                    t2 = time.time() 
                    #text2image
                    t3 = time.time()
                    idx2_all = HammingD(txt_output_test, code_img_all_h, R_h)
                    mApx2 = AQD_new(txt_output_test, C, code_img_all_q, R,l, labelsi,idx2_all)
                    t4 = time.time()
                
                if args.eval_method == 'lossless':
                    t1 = time.time()
                    mApx1 = simple(img_output_test, txt_output, R,l, labelst)
                    t2 = time.time()
                    l=np.copy(test_labels)
                    t3 = time.time()
                    mApx2 = simple(txt_output_test, img_output, R,l, labelsi)
                    t4 = time.time()

                print("test time image2text50"+args.eval_method,t2-t1)
                print("test time text2image50"+args.eval_method,t4-t3)
                
                
                print("test image2text50"+args.eval_method,mApx1)
                print("test text2image50"+args.eval_method,mApx2)
                final_result_test = [mApx1,mApx2]
