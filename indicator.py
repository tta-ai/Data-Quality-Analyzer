import multiprocessing
import xml.etree.ElementTree as elemTree
import glob
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler, TensorDataset
import multiprocessing as mps
from sklearn import preprocessing
from sys import getsizeof
import pickle
import math
import time
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch

import argparse
from dataloader import get_loader
from utils import save_list, load_list
import train
import os
import pickle
import numpy as np


parser = argparse.ArgumentParser(description="LDA calculation")
parser.add_argument('--img', required=True, help='Path of image folder')
parser.add_argument('--meta', required=True, help='Path of meta-data file')
parser.add_argument('--dataset', required=True, help='Dataset name')
parser.add_argument('--process', required=True, help='Process id')
parser.add_argument('--count', required=True, help='Sampling count')
parser.add_argument('--nworkers', required=True, help='The number of core in process')
parser.add_argument('--vector', required=True, help='The number of gaussian vector')
parser.add_argument('--resize', required=True, help='Resize parameter')
parser.add_argument('--ratio', required=True, help='Sampling ratio')
parser.add_argument('--msample', required=True, help='Minimum sampling data per class')


args = parser.parse_args()



if __name__ == '__main__':
    start_time = time.time()
    
    img_folder_path = args.img
    img_meta_path = args.meta
    dataset = args.dataset
    process = int(args.process)
    num_workers = int(args.nworkers)
    normal_vector = int(args.vector)
    resize = int(args.resize)
    ratio = float(args.ratio)
    sampling_count = int(args.count)
    min_sampling_num = int(args.msample)
    
    print('Image folder path = %s\nImage meta path = %s\nDataset name = %s\nProcess num = %d'%(img_folder_path, img_meta_path, 
                                                                               dataset, process))
    print('Num workers = %d\nThe Num of normal vector %d\nResize = 1/%d'%(num_workers, normal_vector, resize))
    print('Sampling ratio : %f\nSampling count = %d\nMinimum sampling num = %d'%(ratio,sampling_count, min_sampling_num))
    
    loader, test_loader = get_loader(img_folder_path, img_meta_path, ratio, sampling_count ,min_sampling_num, num_workers)
    data, label = iter(test_loader).next()
    
    # if RGB image
    if len(data.shape) == 4:
        indicator = train.DatasetQualityEval(loader, process = process, resize = resize, sample_ratio = ratio, sampling_count = sampling_count, normal_vector=normal_vector, batch_size = loader.batch_size, dataset_name = dataset, size = (data.shape[3], data.shape[2], data.shape[1]))
        print('Image shape(width, height) = (%d, %d)\n\n'%(data.shape[3], data.shape[2]))
        print(dataset, 'start!')
    # if Gray image
    elif len(data.shape) == 3:
        indicator = train.DatasetQualityEval(loader, process = process, resize = resize, sample_ratio = ratio, sampling_count = sampling_count, normal_vector=normal_vector, batch_size = loader.batch_size, dataset_name = dataset, size = (data.shape[2], data.shape[1], 1))
        print('Image shape(width, height) = (%d, %d)\n\n'%(data.shape[2], data.shape[1]))
        print(dataset, 'start!')
    
    del data, label    
    
    indicator.coherence(normal_vector)
    
    print(dataset, 'end!')