import os
import random
import psutil
from psutil._common import bytes2human
from collections import OrderedDict
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from ml_analysis.datasets.base_preprocessor import BasePreprocessor
from transformers import BertTokenizerFast

from utils.common import read_file
from ml_analysis.utils import remove_element_names
from ml_analysis.datasets.base_dataset import BaseDataset


class ScreenplayDataset(BaseDataset):
    """
    Dataset for screenplay data ml analysis. Gets screenplays' scenes from annotations made by BERT.
    """

    def __init__(self, tokenizer, config, task, preprocessor):
        super().__init__(tokenizer, config, task, preprocessor)

    def get_screenplay_scenes(self, screenplays):
        screenplay_scenes, text = [], ''
        element_names = [x.name for x in self.config['screenplay_elements']]
        min_scene_length = self.config['tokenization']['nrof_scene_symbols']['min']
        max_scene_length = self.config['tokenization']['nrof_scene_symbols']['max']

        for i, screenplay_text in tqdm(enumerate(screenplays)):
            scenes = []

            for row in screenplay_text.split('\n'):
                if not row.strip():
                    continue
                if row.startswith(self.config['screenplay_elements'].scene_heading.name):
                    if len(text) > min_scene_length and len(text) < max_scene_length:
                        scenes.append(text)
                    text = remove_element_names(element_names, row)
                else:
                    text += remove_element_names(element_names, row)

            if scenes:
                screenplay_scenes.append(scenes)
            else:
                self.imdb_ids.pop(i)

        return screenplay_scenes

    def make_screenplay_generator(self, task_to_label_mapping):
        for file_name in os.listdir(self.config['paths']['screenplay_annotations_dir']):
            file_path = os.path.join(self.config['paths']['screenplay_annotations_dir'], file_name)
            yield read_file(file_path)

    def read_data(self, only_scenes=True):
        task_to_label_mapping = read_file(self.config['paths']['screenplay_labels_mapping_path'])
        self.screenplays, self.imdb_ids = [], []
        screenplays_generator = self.make_screenplay_generator(task_to_label_mapping)

        for file_name in tqdm(os.listdir(self.config['paths']['screenplay_annotations_dir'])):
            imdb_id = file_name.split('_')[1].replace('.txt', '')
            self.imdb_ids.append(imdb_id)
            if not only_scenes:
                self.screenplays.append(next(screenplays_generator))

        if only_scenes:
            self.screenplay_scenes = self.get_screenplay_scenes(screenplays_generator)
        else:
            self.screenplay_scenes = self.get_screenplay_scenes(self.screenplays)

    def get_labels(self):
        task_to_label_mapping = read_file(self.config['paths']['screenplay_labels_mapping_path'])
        self.labels = []

        for i, imdb_id in enumerate(self.imdb_ids.copy()):
            label = task_to_label_mapping[self.task][imdb_id]
            if label == -1:
                self.imdb_ids.pop(i)
                self.screenplays.pop(i)
                self.screenplay_scenes.pop(i)
            else:
                self.labels.append(label)

    def tokenize(self):
        self.inputs, self.attention_masks = [], []

        max_scene_number = self.config['tokenization']['screenplay_dataset']['max_scene_number']
        truncation = self.config['tokenization']['screenplay_dataset']['truncation']
        padding = self.config['tokenization']['screenplay_dataset']['padding']

        for scenes in tqdm(self.screenplay_scenes):
            random_indices = sorted(random.sample(range(len(scenes)), k=min(len(scenes), max_scene_number)))
            scenes = [scene for i, scene in enumerate(scenes) if i in random_indices]

            tokenized = self.tokenizer(scenes, truncation=truncation, padding=padding)

            self.inputs.append(tokenized['input_ids'])
            self.attention_masks.append(tokenized['attention_mask'])

    def to_tensor(self, inputs, masks, labels, imdb_ids):
        inputs = pad_sequence(list(map(torch.tensor, inputs)), padding_value=0, batch_first=True)
        masks = pad_sequence(list(map(torch.tensor, masks)), padding_value=0, batch_first=True)
        labels = torch.LongTensor(labels)
        # TODO: check for nans?
        imdb_ids = torch.LongTensor(list(map(int, imdb_ids)))
        return inputs, masks, labels, imdb_ids

    def split(self, random_state=11):
        test_size = self.config['train']['test_val_split']

        split = train_test_split(self.inputs,
                                 self.attention_masks,
                                 self.labels,
                                 self.imdb_ids,
                                 test_size=test_size,
                                 random_state=random_state,
                                 stratify=self.labels)
        tr_inputs, val_inputs, tr_masks, val_masks, \
        tr_labels, val_labels, tr_imdb_ids, val_imdb_ids = split

        split = train_test_split(val_inputs,
                                 val_masks,
                                 val_labels,
                                 val_imdb_ids,
                                 test_size=0.5,
                                 random_state=random_state)
        tst_inputs, val_inputs, tst_masks, val_masks, \
        tst_labels, val_labels, tst_imdb_ids, val_imdb_ids = split

        self.dataset = {}
        data_keys = ('inputs', 'attention_masks', 'labels', 'imdb_ids')

        self.dataset['train'] = OrderedDict(zip(
            data_keys, self.to_tensor(tr_inputs, tr_masks, tr_labels, tr_imdb_ids)
        ))
        self.dataset['val'] = OrderedDict(zip(
            data_keys, self.to_tensor(val_inputs, val_masks, val_labels, val_imdb_ids)
        ))
        self.dataset['test'] = OrderedDict(zip(
            data_keys, self.to_tensor(tst_inputs, tst_masks, tst_labels, tst_imdb_ids)
        ))

    def prepare_dataset(self):
        self.read_data(only_scenes=True)
        self.get_labels()
        self.tokenize()
        self.split()


if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    preprocessor = BasePreprocessor()
    from ml_analysis.config import MovieDataConfig

    SD = ScreenplayDataset(tokenizer, MovieDataConfig, 'movie_script_awards', preprocessor)
    SD.prepare_dataset()
