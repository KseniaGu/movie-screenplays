import os
from collections import OrderedDict
import torch
from torch.nn.utils.rnn import pad_sequence
from utils.common import read_file, write_file
from sklearn.model_selection import train_test_split
from ml_analysis.ml_utils import remove_element_names
from transformers import BertTokenizerFast
from ml_analysis.datasets.base_preprocessor import BasePreprocessor
from ml_analysis.datasets.base_dataset import BaseDataset


class CharacterDataset(BaseDataset):
    """
    Dataset for movie characters data ml analysis.
    """

    def __init__(self, tokenizer, config, task, preprocessor):
        super().__init__(tokenizer, config, task, preprocessor)

    def get_classes_mapping(self):
        if self.task == self.config['tasks'].character_gender.name:
            return {0: 'male', 1: 'female'}
        elif self.task == self.config['tasks'].character_role.name:
            return {0: 'minor', 1: 'main'}
        else:
            return {}

    def read_imdb_chars_texts(self):
        imdb_chars_texts = {}

        for movie_name in os.listdir(self.config['paths']['character_texts_dir']):
            imdb_id = movie_name.split('_')[1].replace('.txt', '')

            if not imdb_id in imdb_chars_texts:
                imdb_chars_texts[imdb_id] = OrderedDict()
                movie_dir = os.path.join(self.config['paths']['character_texts_dir'], movie_name)
                for character_name in os.listdir(movie_dir):
                    character_text_file = os.path.join(movie_dir, character_name)
                    imdb_chars_texts[imdb_id][character_name.split('_')[0]] = read_file(character_text_file)

        return imdb_chars_texts

    def read_data(self):
        self.imdb_chars_to_labels = read_file(os.path.join(
            self.config['paths']['character_labels_dir'], self.task + '_labels.pickle'))
        self.imdb_chars_texts = self.read_imdb_chars_texts()

    def tokenize(self, to_load=False):
        self.input_ids, self.attention_masks, self.labels, self.char_ids = [], [], [], []

        if to_load:
            self.imdb_chars_train_data = read_file(self.config['paths']['imdb_chars_train_data'])
            self.char_ids = list(self.imdb_chars_train_data.keys())

            for data in self.imdb_chars_train_data.values():
                self.input_ids.append(data['input_ids'])
                self.attention_masks.append(data['attention_masks'])
                self.labels.append(data['label'])
        else:
            self.imdb_chars_train_data = {}
            element_names = [x.name for x in self.config['screenplay_elements']]

            for imdb_id, imdb_chars_text in self.imdb_chars_texts.items():
                if not imdb_id in self.imdb_chars_to_labels:
                    continue
                for char_idx, (imdb_char, text) in enumerate(self.imdb_chars_texts[imdb_id].items()):
                    if not imdb_char in self.imdb_chars_to_labels[imdb_id]:
                        continue

                    text = ' '.join([remove_element_names(element_names, x) for x in text.split('\n')])
                    text = self.preprocessor.preprocess_base(text)

                    encoded_dict = self.tokenizer.encode_plus(
                        text, add_special_tokens=True, pad_to_max_length=False, return_attention_mask=True)

                    label = self.imdb_chars_to_labels[imdb_id][imdb_char]
                    self.imdb_chars_train_data[imdb_char + '_' + str(imdb_id)] = {
                        'input_ids': encoded_dict['input_ids'],
                        'attention_masks': encoded_dict['attention_mask'],
                        'label': label
                    }

                    self.input_ids.append(encoded_dict['input_ids'])
                    self.attention_masks.append(encoded_dict['attention_mask'])
                    self.char_ids.append((char_idx, int(imdb_id)))
                    self.labels.append(label)

            write_file(self.imdb_chars_train_data, self.config['paths']['imdb_chars_train_data'])

    def to_tensor(self, inputs, attention_masks, labels, char_ids):
        max_len = self.config['tokenization']['character_dataset']['max_seq_length']
        inputs = pad_sequence(list(map(lambda x: torch.tensor(x[:max_len]), inputs)),
                              padding_value=0, batch_first=True)
        attention_masks = pad_sequence(list(map(lambda x: torch.tensor(x[:max_len]), attention_masks)),
                                       padding_value=0, batch_first=True)
        labels = torch.LongTensor(labels)
        char_ids = torch.LongTensor(char_ids)
        return inputs, attention_masks, labels, char_ids

    def split(self, random_state=11):
        test_size = self.config['train']['test_val_split']

        split = train_test_split(self.input_ids,
                                 self.attention_masks,
                                 self.labels,
                                 self.char_ids,
                                 test_size=test_size,
                                 random_state=random_state,
                                 stratify=self.labels)

        tr_inputs, val_inputs, tr_masks, val_masks, \
        tr_labels, val_labels, tr_char_ids, val_char_ids = split

        split = train_test_split(val_inputs,
                                 val_masks,
                                 val_labels,
                                 val_char_ids,
                                 test_size=0.5,
                                 random_state=random_state)
        tst_inputs, val_inputs, tst_masks, val_masks, \
        tst_labels, val_labels, tst_char_ids, val_char_ids = split

        self.dataset = {}
        data_keys = ('inputs', 'attention_masks', 'labels', 'char_ids')

        self.dataset['train'] = OrderedDict(zip(
            data_keys, self.to_tensor(tr_inputs, tr_masks, tr_labels, tr_char_ids)
        ))
        self.dataset['val'] = OrderedDict(zip(
            data_keys, self.to_tensor(val_inputs, val_masks, val_labels, val_char_ids)
        ))
        self.dataset['test'] = OrderedDict(zip(
            data_keys, self.to_tensor(tst_inputs, tst_masks, tst_labels, tst_char_ids)
        ))

    def prepare_dataset(self):
        self.read_data()
        self.tokenize()
        self.split()
