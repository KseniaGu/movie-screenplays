import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, tokenizer, config, task, preprocessor):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.task = task
        self.phase = 'train'
        self.preprocessor = preprocessor

    def set_phase(self, phase='train'):
        self.phase = phase

    def __len__(self):
        return len(self.dataset[self.phase]['labels'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        data = {}
        for key in self.dataset[self.phase].keys():
            data[key] = self.dataset[self.phase][key][idx]

        return data
