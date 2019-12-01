import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler, TensorDataset
from PIL import Image
from torchvision import transforms
import pickle
import cv2


class DSET(Dataset):
    def __init__(self, image_path_list, label_list):
        self.image_path_list = image_path_list
        self.label_list = label_list
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, idx):
        path = self.image_path_list[idx]
        img = Image.open(path)
        img2 = np.array(img)
        img.close()
        
        if img2.ndim != 3:
            img2 = cv2.cvtColor(img2.astype('uint8'), cv2.COLOR_GRAY2RGB)
        
        img2 = Image.fromarray(img2)
        transformed_img = self.transform(img2)
        label = self.label_list[idx]
        img2.close()
        
        return transformed_img ,label
    
    
class StratifiedSampler(Sampler):
    """Stratified Sampling

    Provides equal representation of target classes in each batch
    """
    
    # self.label_dict : { class : [idx] }
    # self.class_instance_dict : { class : 데이터 수(밖에서 미리 전처리해서 들어옴, 전체 데이터수가 30개 미만이면 0 등등) }
    # self.batch_size : batch_sampling_number
    # self.min_class_data_size : 클래스내 최소 데이터 수
    def __init__(self, label_dict, sampling_count, class_instance_dict, batch_size, min_class_data_size):
        self.label_dict = label_dict
        self.class_instance_dict = class_instance_dict
        self.batch_size = batch_size
        self.sampling_count = sampling_count
        self.min_class_data_size = min_class_data_size
        
    def gen_sample_idx(self):
        sample_idx = []
        
        # sampling count 수 만큼 iteration을 돔 (300으로 하기로 함)
        for _ in range(self.sampling_count):
            tmp_idx = []
            
            for label in self.label_dict.keys():
                # 전체 데이터 수 가 30개 보다 적은 경우
                if self.class_instance_dict[label] == 0:
                    continue
                else:
                    stratified_idx = np.random.choice(self.label_dict[label], self.class_instance_dict[label])
                    tmp_idx.extend(stratified_idx)
                
            sample_idx.append(tmp_idx)
        
        print('Sampler index size :', len(tmp_idx))
        return np.array(sample_idx).reshape(-1)
        
        
    def __iter__(self):
        return iter(self.gen_sample_idx())

    def __len__(self):
        return len(self.class_vector)

    
    
def get_loader(img_folder_path, img_meta_path, sample_ratio, sampling_count ,min_sampling_num, num_workers):

    """
    img_folder_path : path of image folder
    
    img_meta_path : path of image csv file
    
    sample_ratio : sampling ratio
    
    sampling_count = sampling count
    
    min_sampleing_num = minimum sampling number
    
    num_workers = # of process for dataloader
    
    """

    df = pd.read_csv(img_meta_path)
    
    #이미지 path 및 label
    image_path_list = [img_folder_path + path for path in df['ImageFileName']]
    label_list = df['Label']
    
    # 사용할 배치사이즈 설정
    standard_batch_size = int(len(label_list) * sample_ratio)
    
    #sampling할 데이터 수 정함, 데이터셋의 크기가 크면 10000개로 고정
    if standard_batch_size > 10000:
        print('Data size is too large --> 10000')
        standard_batch_size = 10000
    
    # label_count_dict : {class : class에 해당하는 데이터 수}
    unique_label = list(set(label_list))
    label_count_dict = {key : 0 for key in unique_label}
    for l in label_list:
        label_count_dict[l] += 1

    print('Total Data size = ', len(label_list))
    
    # label_idx_dict : {class : class에 해당하는 Label의 idx}
    label_idx_dict = {key : [] for key in unique_label}
    for idx, l in enumerate(label_list):
        label_idx_dict[l].append(idx)
    
    
    # label_stratified_sampling_num : {class : 해당 클래스에 대해 sampling된 data수}
    label_stratified_sampling_num = {}
    
    for key, val in label_count_dict.items():
        label_stratified_sampling_num[key] = 0
        
    batch_sampling_num = 0
    total_data_num = len(label_list)
    
    for key, val in label_count_dict.items():
        # 전체 데이터에 대한 해당 class data수의 비율만큼 batch_size를 가져옴, 결국 sample ratio만큼 가져오는 것과 동일함
        part = round(val / total_data_num * standard_batch_size)
        
        # 해당 클래스의 데이터 수가 min_sampling_num보다 작은 경우 학습에서 제외
        if val < min_sampling_num :
            print("Not enough data point : ", key, 'deleted')
            label_stratified_sampling_num[key] = 0
            continue
            
        # 해당 클래스의 데이터 수가 min_sampling_num 보다는 많지만 sampling한 데이터의 수가 min_sampling_num보다 작은경우 min_sampling_num개를 가져옴
        elif part < min_sampling_num:
            label_stratified_sampling_num[key] = min_sampling_num
            batch_sampling_num += min_sampling_num
        else:
            label_stratified_sampling_num[key] = part
            batch_sampling_num += part
    print('Batch sampling num :', batch_sampling_num)
    
    sampler = StratifiedSampler(label_dict=label_idx_dict,
                            sampling_count = sampling_count,
                            class_instance_dict = label_stratified_sampling_num,
                            batch_size = batch_sampling_num,
                            min_class_data_size = min_sampling_num
                            )

    dataset = DSET(image_path_list, label_list)
    loader = DataLoader(dataset, batch_size=batch_sampling_num, shuffle=False, num_workers=num_workers, sampler=sampler)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, sampler=sampler)
    
    return loader, test_loader