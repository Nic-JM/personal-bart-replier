from torch.utils.data import Dataset

class EncodedData(Dataset):
    def __init__(self, processed_data):
        self.data = processed_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
