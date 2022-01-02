from collections import Counter

import torch
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


class BaseDataloader:
    def __init__(self, training_config):
        self.config = training_config

    def make_weights(self, labels):
        if torch.is_tensor(labels):
            labels = labels.tolist()

        count = dict(Counter(labels))
        N = float(sum(count.values()))

        task = self.config['task']
        nrof_classes = len(self.config['classes_names'][task])
        weight_per_class = {}
        for i in range(nrof_classes):
            if i in count:
                weight_per_class[i] = (N / float(count[i]))
            else:
                weight_per_class[i] = 1000

        weight = [weight_per_class[label] for label in labels]

        return weight, weight_per_class

    def get_dataloader(self, dataset, sampler=None, phase='train', to_make_weights=False):
        if phase == 'train' and sampler is None and not to_make_weights:
            sampler = RandomSampler(dataset)
        elif to_make_weights:
            weights, weight_per_class = self.make_weights(dataset.dataset['train']['labels'])
            sampler = WeightedRandomSampler(weights, len(weights))

        dataloader = DataLoader(dataset,
                                sampler=sampler,
                                batch_size=self.config['batch_size'][phase],
                                drop_last=phase == 'train')
        return dataloader


if __name__ == '__main__':
    ...
