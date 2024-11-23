import torch  
from torch.utils.data import Dataset  

class TwoStreamDataset(Dataset):  
    def __init__(self, num_samples=1000):  
        self.num_samples = num_samples  
        self.flow_data = torch.rand(num_samples, 8, 32, 32)
        self.rgb_data = torch.rand(num_samples, 3, 32, 32) 
        self.labels = torch.randint(0, 10, (num_samples,)) 

    def __len__(self):  
        return self.num_samples  

    def __getitem__(self, idx):  
        return {  
            'flow': self.flow_data[idx],  
            'rgb': self.rgb_data[idx],  
            'label': self.labels[idx]  
        }  
