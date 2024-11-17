from torch.utils.data import Sampler
import torch
from tqdm import tqdm

class FilterNoneSampler(Sampler):
    def __init__(self, dataset, shuffle=False):
        """
        过滤返回None的数据的采样器
        
        Args:
            dataset: 数据集
            shuffle: 是否打乱数据
        """
        self.dataset = dataset
        self.shuffle = shuffle
        # 过滤掉返回None的索引
        self.indices = []
        for i in tqdm(range(len(dataset))):
            item = dataset[i]
            if item is not None:
                self.indices.append(i)
                
    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.indices)).tolist()
            return iter(self.indices[i] for i in indices)
        return iter(self.indices)
        
    def __len__(self):
        return len(self.indices)