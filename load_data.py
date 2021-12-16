import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils import data
from PIL import Image


class MM_pre_feature_Dataset(data.Dataset):
    '''
    Multi-modal retrieval data loader.
    '''
    def __init__(self, image_feature, text_feature, img_labels, txt_labels, transform=None, target_transform=None):
        'Initialization'
        self.img_l = img_labels
        self.txt_l = txt_labels
        image_feature = np.swapaxes(image_feature,1,3)
        self.image_feature = image_feature
        self.transform = transform
        self.text_feature = text_feature

    def __len__(self):
        return len(self.img_l)

    def __getitem__(self, idx):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        
        image = self.image_feature[idx]
        text = self.text_feature[idx]
        img_l = self.img_l[idx]
        txt_l = self.txt_l[idx]

        if self.transform is not None:
            image = self.transform(image)
            
        return {'image': image,'text':text, 'img_l':img_l, 'txt_l': txt_l},



def load_dataset(args):
    '''
    load_dataset download and process the dataset into train/val/test
    args:
        config: defined in the main.py

    '''
    if args.dataset == 'mirflickr':
        print('loading mirflickr data...')
        with open('./data/mir/vali_text.npy', 'rb') as f:
            vali_text = np.load(f)
        with open('./data/mir/vali_image.npy', 'rb') as f:
            vali_image = np.load(f)
        with open('./data/mir/vali_label.npy', 'rb') as f:
            vali_label = np.load(f)

        with open('./data/mir/test_text.npy', 'rb') as f:
            test_text = np.load(f)
        with open('./data/mir/test_image.npy', 'rb') as f:
            test_image = np.load(f)
        with open('./data/mir/test_label.npy', 'rb') as f:
            test_label = np.load(f)

        with open('./data/mir/train_text.npy', 'rb') as f:
            train_text = np.load(f)
        with open('./data/mir/train_image.npy', 'rb') as f:
            train_image = np.load(f)
        with open('./data/mir/train_text_label.npy', 'rb') as f:
            train_text_label = np.load(f)
        with open('./data/mir/train_image_label.npy', 'rb') as f:
            train_img_label = np.load(f)
        '''
        texts = np.load('/yang_data/mirflickr25k_images/mirflickr/meta/tags/tags_bag_of_word_1386.npy')
        labels = np.load('/yang_data/mirflickr25k_annotations/24_label_all.npy')
        images = np.load('/yang_data/mirflickr25k_images/mirflickr/norm_raw_img_features_all.npy')
        

        train_n = 10000
        test_n = 2000
        vali_n = 2000

        train_text = texts[0:train_n,:]
        test_text = texts[train_n:train_n+test_n,:]


        train_image = images[0:train_n]
        test_image = images[train_n:train_n+test_n]

        train_label = labels[0:train_n,:]
        test_labels = labels[train_n:train_n+test_n,:]


        a = np.arange(train_n)
        np.random.seed(10)
        np.random.shuffle(a)
        train_img_label = np.copy(train_label)
        train_text = train_text[a,:]
        train_text_label = train_label[a,:]

        total_sim = np.int8(np.sum(train_img_label*train_text_label,axis=1)>0)
        ts = np.count_nonzero(total_sim)
        print("training similar percentage:", ts/train_n)

        vali_n = 2000
        vali_text = texts[25000-vali_n:25000,:]
        vali_image = images[25000-vali_n:25000]
        vali_label = labels[25000-vali_n:25000,:]
        '''
        
    if args.dataset == 'nuswide':
        print('loading nuswide data...')

        with open('./data/nus/vali_text.npy', 'rb') as f:
            vali_text = np.load(f)
        with open('./data/nus/vali_image.npy', 'rb') as f:
            vali_image = np.load(f)
        with open('./data/nus/vali_label.npy', 'rb') as f:
            vali_label = np.load(f)

        with open('./data/nus/test_text.npy', 'rb') as f:
            test_text = np.load(f)
        with open('./data/nus/test_image.npy', 'rb') as f:
            test_image = np.load(f)
        with open('./data/nus/test_label.npy', 'rb') as f:
            test_label = np.load(f)

        with open('./data/nus/train_text.npy', 'rb') as f:
            train_text = np.load(f)
        with open('./data/nus/train_image.npy', 'rb') as f:
            train_image = np.load(f)
        with open('./data/nus/train_text_label.npy', 'rb') as f:
            train_text_label = np.load(f)
        with open('./data/nus/train_image_label.npy', 'rb') as f:
            train_img_label = np.load(f)
        '''
        idx_list_vali = np.load('../nuswide/idx_list_vali.npy')
        idx_list_train = np.load('../nuswide/idx_list_train.npy')
        idx_list_test = np.load('../nuswide/idx_list_test.npy')
        train_image = np.load('../nuswide/norm_raw_img_features_train.npy')
        test_image = np.load('../nuswide/norm_raw_img_features_test.npy')
        vali_image = np.load('../nuswide/norm_raw_img_features_vali.npy')

        text = np.load('../nuswide/tags_occurance_1000.npy')
        labels = np.load('../nuswide/new_label.npy')

        a = np.arange(195834)
        np.random.seed(10)
        np.random.shuffle(a)

        train_text = text[a[idx_list_train-1],:]
        test_text = text[a[idx_list_test-1],:]
        vali_text = text[a[idx_list_vali-1],:]
        train_labels = labels[a[idx_list_train-1],:]
        test_labels = labels[a[idx_list_test-1],:]
        vali_label = labels[a[idx_list_vali-1],:]


        train_n = 10500
        test_n = 2100
        vali_n = 1050
        a = np.arange(train_n)
        np.random.seed(10)
        np.random.shuffle(a)
        train_img_label = np.copy(train_labels)
        train_text = train_text[a,:]
        train_text_label = np.copy(train_labels[a,:])


        total_sim = np.int8(np.sum(train_img_label*train_text_label,axis=1)>0)
        ts = np.count_nonzero(total_sim)
        print("training similar percentage:", ts/train_n)
        
        
        #reprocess nus-wide text feature: remove freq. and use 0 or 1
        for i in range(train_text.shape[0]):
            for j in range(train_text.shape[1]):
                if train_text[i][j] > 0:
                    train_text[i][j] = 1
        for i in range(test_text.shape[0]):
            for j in range(test_text.shape[1]):
                if test_text[i][j] > 0:
                    test_text[i][j] = 1
        for i in range(vali_text.shape[0]):
            for j in range(vali_text.shape[1]):
                if vali_text[i][j] > 0:
                    vali_text[i][j] = 1
        '''
    img = (train_image,vali_image,test_image)
    txt = (train_text,vali_text,test_text)
    labels = (train_text_label,train_img_label,vali_label,test_label)
    return img, txt, labels
