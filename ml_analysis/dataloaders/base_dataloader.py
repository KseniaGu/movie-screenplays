from collections import Counter

import torch
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from ml_analysis.datasets.character_dataset import CharacterDataset
from ml_analysis.datasets.screenplay_dataset import ScreenplayDataset
from ml_analysis.datasets.next_scene_dataset import NextSceneDataset
from ml_analysis.datasets.base_preprocessor import BasePreprocessor


class BaseDataloader:
    def __init__(self, training_config):
        self.config = training_config

    def make_weights(self, labels):
        if torch.is_tensor(labels):
            labels = labels.tolist()

        count = dict(Counter(labels))
        N = float(sum(count.values()))

        weight_per_class = {}
        for i in range(self.config['num_classes']):
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
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    preprocessor = BasePreprocessor()
    from ml_analysis.config import MovieDataConfig
    import psutil
    from psutil._common import bytes2human

    # print(bytes2human(psutil.virtual_memory().available))
    # print('before SD', bytes2human(psutil.virtual_memory().used))

    NSD = NextSceneDataset(tokenizer, MovieDataConfig, 'next_scene_prediction', preprocessor)
    NSD.prepare_dataset()
    # SD = ScreenplayDataset(tokenizer, MovieDataConfig, 'movie_script_awards', preprocessor)
    # SD.prepare_dataset()
    # CD = CharacterDataset(tokenizer, MovieDataConfig, 'character_gender', preprocessor)
    # CD.prepare_dataset()

    # print('after SD', bytes2human(psutil.virtual_memory().used))
    # SD.read_data(only_scenes=True)
    # CD.read_data()
    # print('after read', bytes2human(psutil.virtual_memory().used))
    # print(bytes2human(psutil.virtual_memory().available))
    # SD.prepare_scenes()
    # CD.tokenize()
    # print('after prepare', bytes2human(psutil.virtual_memory().used))
    # print(bytes2human(psutil.virtual_memory().available))
    # SD.split()
    # CD.split()
    SDL = BaseDataloader(MovieDataConfig['train'])

    # SD.set_phase('train')
    # CD.set_phase('train')
    NSD.set_phase('train')

    # train_dataloader = SDL.get_dataloader(SD, phase='train', to_make_weights=True)
    # train_dataloader = SDL.get_dataloader(CD, phase='train', to_make_weights=True)
    train_dataloader = SDL.get_dataloader(NSD, phase='train', to_make_weights=False)
    for d in train_dataloader:
        print(d)
        break
    print('after train loader', bytes2human(psutil.virtual_memory().used))
    print(bytes2human(psutil.virtual_memory().available))

    # SD.set_phase('val')
    # val_dataloader = SDL.get_dataloader(SD, phase='val', to_make_weights=False)
    # SD.set_phase('test')
    # test_dataloader = SDL.get_dataloader(SD, phase='test', to_make_weights=False)
