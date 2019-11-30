import numpy as np
import random
import time
import os
from sklearn import preprocessing
import pandas as pd
import sys
import smtplib

import math
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, TensorDataset
from PIL import Image
from torchvision import transforms
import pickle
import pandas as pd
from datetime import datetime

class DatasetQualityEval():
    # MNIST : 1000 x 784 ( data개수 x flatten features)
    # numpy array를 input으로 받음.
    def __init__(self, loader, process= 1 ,resize = 1, sample_ratio = 0.3, sampling_count = 2, 
                 normal_vector = 10, batch_size = 100, dataset_name = 'NoName', size = (256, 256, 3)):
        # data 
        self.loader = loader
        
        self.size = size
        self.resize = resize
        self.process = process
        
        # data를 sampling할 비율, 횟수 (bootstrap)
        self.sample_ratio =  sample_ratio
        self.sampling_count = sampling_count
        
        self.dset_name = dataset_name
        
        # (data 개수 x data 차원)을 입력받는 가정. (image를 가로로 늘린 데이터)
        self.data_dim = size[0] * size[1] * size[2]
        
        # coherence (LDA)
        self.coherence_result = 0
        self.avg_Sb = 0
        self.avg_Sw = 0
        
        self.normal_vec_num = normal_vector
        self.between_vect_mean = {}
        
        self.LDA_list = []
        self.SB_list = []
        self.SW_list = []
        self.S_B_variance_list = []
        self.batch_size = batch_size
        
        
        
    def cal_lda(self, sampled_X_data, sampled_Y_data):
        print("Start calculating")
        sampled_X_data = sampled_X_data.astype('float32')
        scaler = preprocessing.StandardScaler()
        sampled_X_data = scaler.fit_transform(sampled_X_data)
        add = {}
        count = {}
        S_B_variance = {}
        S_B_list = []
        S_W_list = []
        LDA_list = []
        for idx in range(sampled_Y_data.shape[0]):
            label = sampled_Y_data[idx]
            count[label] = 1 if label not in list(count.keys()) else count[label] + 1
            add[label] = sampled_X_data[idx] if label not in list(add.keys()) else add[label] + sampled_X_data[idx]
        each_class_mean = {each_class : (add[label]/count[label]).reshape(self.data_dim, 1) for each_class in list(add.keys())}
        global_mean = np.mean(sampled_X_data, axis=0).reshape(self.data_dim, 1)
        
        gaussian_vec = np.random.RandomState().normal(0, 1, (self.normal_vec_num, self.data_dim))
        max_lda, max_s_b, max_s_w, max_gaussian = 0, 0, 0, 0
        
        # matrix로 안하는게 좋을듯. 
        # (# class x data_dim) matrix 만들다가 memory 터질수도 있음.
        # class 하나씩 계산하는게 나을듯.
        for one_gaussian_vec in gaussian_vec:
            S_B = 0
            S_W = 0
            for label, mean_vec in each_class_mean.items():
                n = count[label]
                # between class 
                between_class = (n / sampled_Y_data.shape[0]) * np.matmul(one_gaussian_vec.T, (mean_vec - global_mean)) * np.matmul((mean_vec - global_mean).T , one_gaussian_vec)
                # dimension으로 나눠줌.
                between_class /= self.data_dim
                #variance[label] = between_class if label not in list(variance.keys()) else variance[label] + between_class
                S_B += between_class
                # within class
                label_instance = sampled_X_data[np.where(sampled_Y_data == label)]
                within_class = np.matmul( np.matmul(one_gaussian_vec.T , (label_instance - mean_vec.T).T), np.matmul( (label_instance - mean_vec.T), one_gaussian_vec)) / n
                # dimension으로 나눠줌.
                within_class /= self.data_dim
                S_W += within_class
                S_B_variance[label] = within_class if label not in S_B_variance.keys() else S_B_variance[label] + within_class
                
            
            S_B /= len(each_class_mean.items())
            S_W /= len(each_class_mean.items())
            S_W_list.append(S_W.item())
            S_B_list.append(S_B.item())
            # 모든 LDA 저장
            lda = S_B / S_W
            LDA_list.append(lda.item())
            
            dtime = datetime.fromtimestamp(time.time())
            f_sb = open('./%s_resize%d_ratio%f_count%d_gvn%d_SB_log.txt'%(self.dset_name, self.resize, self.sample_ratio, self.sampling_count, self.normal_vec_num), mode='a+', encoding='utf-8')
            f_sw = open('./%s_resize%d_ratio%f_count%d_gvn%d_SW_log.txt'%(self.dset_name, self.resize, self.sample_ratio, self.sampling_count, self.normal_vec_num), mode='a+', encoding='utf-8')
            f_lda = open('./%s_resize%d_ratio%f_count%d_gvn%d_lda_log.txt'%(self.dset_name, self.resize, self.sample_ratio, self.sampling_count, self.normal_vec_num), mode='a+', encoding='utf-8')
            f_sb.write('%s_%d\t%s\n' % (dtime, self.process, S_B.item())) 
            f_sw.write('%s_%d\t%s\n' % (dtime, self.process, S_W.item()))
            f_lda.write('%s_%d\t%s\n' % (dtime, self.process, lda.item()))
            f_sb.close()
            f_sw.close()
            f_lda.close()
            
            if lda > max_lda:
                max_lda = lda
                max_s_b = S_B
                max_s_w = S_W
                max_gaussian = one_gaussian_vec

        S_B_variance = {label: (within_vect / (count[label]*self.normal_vec_num)) for label, within_vect in S_B_variance.items()}
        f_sb_variance = open('./%s_resize%d_ratio%f_count%d_gvn%d_SB_variance.txt'%(self.dset_name, self.resize, self.sample_ratio, self.sampling_count, self.normal_vec_num), mode='a+', encoding='utf-8')
        f_sb_variance.write('%s_%d\t%s\n' % (dtime, self.process, S_B_variance))
        f_sb_variance.close()
        
        
        return S_B_variance, S_W_list, S_B_list, LDA_list
    
    
    def coherence(self, normal_vec_num = 100):
        start_time = time.time()
        self.normal_vec_num = normal_vec_num

        for data, label in self.loader:
            iter_start_time = time.time()
            data = np.array(data).reshape(self.batch_size, -1)
            label = np.array(label)
            S_B_variance, SW, SB, LDA = self.cal_lda(data, label)
            self.S_B_variance_list.append(S_B_variance)
            self.SW_list.extend(SW)
            self.SB_list.extend(SB)
            self.LDA_list.extend(LDA)
            iter_end_time = time.time()
            print('Iter time = %.4f' % (iter_end_time - iter_start_time))
        
        # coherence 값을 mean 값으로 바꿈.
        self.coherence_result = np.mean(self.LDA_list)
        
        
        end_time = time.time()
        elapsed_time = end_time-start_time
        
        print("\nSample ratio: %.2f, Sampling count: %d" %(self.sample_ratio, self.sampling_count))
        print("Data coherence:", self.coherence_result)
        print("Computing time: %d hour %d min %d sec (%.3f)\n" % (elapsed_time/3600, (elapsed_time%3600)/60, elapsed_time%60, elapsed_time))
        f = open('./%s_process%d_resize%d_ratio%f_count%d_gvn%d.txt'%(self.dset_name, self.process, self.resize, self.sample_ratio, self.sampling_count, self.normal_vec_num), mode='wt', encoding='utf-8')
        f.write('resize : 1/%d\n' % (self.resize))
        f.write('Sampled data : %d\n' % (self.batch_size) )
        f.write("Sample ratio: %.2f, Sampling count: %d\n" %(self.sample_ratio, self.sampling_count))
        f.write("Num Normal vector : %d \n" % (self.normal_vec_num))
        f.write("Data coherence : %.8f\n" % self.coherence_result)
        
        f.write("Computing time: %d hour %d min %d sec (%.3f)\n" % (elapsed_time/3600, (elapsed_time%3600)/60, elapsed_time%60, elapsed_time))
        f.close()
        
        return self.coherence_result
        
    def between_class_mean(self):
        class_labels = list(set(j  for i in self.S_B_variance_list for j in i.keys()))
        class_count = {}
        add_vec = {}
        for label in class_labels:
            for i in range(len(self.S_B_variance_list)):
                if label in (self.S_B_variance_list[i].keys()):
                    class_count[label] = 1 if label not in class_count.keys() else class_count[label] + 1
                    add_vec[label] = self.S_B_variance_list[i][label] if label not in add_vec.keys() else add_vec[label] + self.S_B_variance_list[i][label]
        
        self.between_vect_mean = {label : (add / float(class_count[label])) for label, add in add_vec.items()}
        return self.between_vect_mean
        
    

