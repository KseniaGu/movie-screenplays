import random
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from tqdm import tqdm

from ml_analysis.datasets.screenplay_dataset import ScreenplayDataset


class NextSceneDataset(ScreenplayDataset):
    """
    Dataset for Next Scene Prediction (NScP) task.
    """

    def __init__(self, tokenizer, config, task, preprocessor):
        super().__init__(tokenizer, config, task, preprocessor)

    def add_negative_example(self, i, screenplay, scene_pairs, labels):
        """
        Adds pairs of scenes:
            1. from different (not neighbouring) parts of the same screenplay,
            2. from different screenplays.

        Args:
            i (int): scene order number (within <screenplay>)
            screenplay (str): currently observed screenplay
            scene_pairs (list): list of already added scene pairs
            labels (list): list of already added labels

        Returns: updated <scene_pairs>, <labels>
        """
        another_screenplay = random.choice((True, False))

        if i + 2 < len(screenplay) and not another_screenplay:
            random_scene_idx = random.choice(list(range(i - 1)) + list(range(i + 2, len(screenplay))))

            # keep order making negative examples harder
            if random_scene_idx > i:
                scene_pairs.append([screenplay[i], screenplay[random_scene_idx]])
            else:
                scene_pairs.append([screenplay[random_scene_idx], screenplay[i]])
        else:
            another_screenplay = random.choice([x for x in self.screenplay_scenes if x != screenplay])
            random_scene_idx = random.choice(list(range(len(another_screenplay))))

            if random.choice((True, False)):
                scene_pairs.append([screenplay[i], another_screenplay[random_scene_idx]])
            else:
                scene_pairs.append([another_screenplay[random_scene_idx], screenplay[i]])
        labels.append(0)

        return scene_pairs, labels

    def make_scene_pairs_and_labels(self):
        """Makes positive and negative examples of scene pairs."""
        self.scene_pairs, self.labels = [], []

        for screenplay in tqdm(self.screenplay_scenes):
            for i in range(1, len(screenplay)):
                self.scene_pairs.append([screenplay[i - 1], screenplay[i]])
                self.labels.append(1)

                self.scene_pairs, self.labels = self.add_negative_example(i, screenplay, self.scene_pairs, self.labels)

    def tokenize(self):
        """Tokenizes, numericalizes, makes attention masks and token type ids."""
        self.inputs, self.token_type_ids, self.attention_masks = [], [], []

        max_nrof_examples = self.config['tokenization']['next_scene_dataset']['max_nrof_pairs']
        scene_pairs_to_tokenize = self.scene_pairs[:max_nrof_examples]
        self.labels = self.labels[:max_nrof_examples]

        max_length = self.config['tokenization']['next_scene_dataset']['max_seq_length']
        truncation = self.config['tokenization']['next_scene_dataset']['truncation']
        padding = self.config['tokenization']['next_scene_dataset']['padding']

        tokenized = self.tokenizer(scene_pairs_to_tokenize,
                                   max_length=max_length,
                                   truncation=truncation,
                                   padding=padding,
                                   return_tensors='pt')

        self.inputs = tokenized['input_ids']
        self.token_type_ids = tokenized['token_type_ids']
        self.attention_masks = tokenized['attention_mask']

    def split(self, random_state=11):
        test_size = self.config['train']['test_val_split']

        split = train_test_split(self.inputs,
                                 self.attention_masks,
                                 self.labels,
                                 self.token_type_ids,
                                 test_size=test_size,
                                 random_state=random_state)

        tr_inputs, val_inputs, tr_masks, val_masks, \
        tr_labels, val_labels, tr_token_type_ids, val_token_type_ids = split

        split = train_test_split(val_inputs,
                                 val_masks,
                                 val_labels,
                                 val_token_type_ids,
                                 test_size=0.5,
                                 random_state=random_state)
        tst_inputs, val_inputs, tst_masks, val_masks, \
        tst_labels, val_labels, tst_token_type_ids, val_token_type_ids = split

        self.dataset = {}
        data_keys = ('inputs', 'attention_masks', 'labels', 'token_type_ids')

        self.dataset['train'] = OrderedDict(zip(
            data_keys, (tr_inputs, tr_masks, tr_labels, tr_token_type_ids)
        ))
        self.dataset['val'] = OrderedDict(zip(
            data_keys, (val_inputs, val_masks, val_labels, val_token_type_ids)
        ))
        self.dataset['test'] = OrderedDict(zip(
            data_keys, (tst_inputs, tst_masks, tst_labels, tst_token_type_ids)
        ))

    def prepare_dataset(self):
        self.read_data(only_scenes=True)
        self.make_scene_pairs_and_labels()
        self.tokenize()
        self.split()


if __name__ == '__main__':
    import psutil
    from psutil._common import bytes2human
    from ml_analysis.datasets.base_preprocessor import BasePreprocessor
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    preprocessor = BasePreprocessor()
    from ml_analysis.config import MovieDataConfig

    NSD = NextSceneDataset(tokenizer, MovieDataConfig, 'next_scene_prediction', preprocessor)
    NSD.prepare_dataset()
