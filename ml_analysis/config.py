from enum import Enum
import torch
from transformers import BertTokenizerFast, BertModel

MovieDataConfig = {
    'paths':
        {
            'screenplays_dir': '/home/ksen/PycharmProjects/movie-screenplays/data/movie_screenplay_dataset/screenplay_data/raw_texts',
            'screenplay_annotations_dir': '/home/ksen/PycharmProjects/movie-screenplays/data/movie_screenplay_dataset/screenplay_data/BERT_annotations',
            'screenplay_labels_mapping_path': '/home/ksen/PycharmProjects/movie-screenplays/data/movie_screenplay_dataset/screenplay_data/script_task_to_labels_dicts.pickle',
            'character_texts_dir': '/home/ksen/PycharmProjects/movie-screenplays/data/movie_screenplay_dataset/movie_characters/bert_anno_character_texts',
            'character_labels_dir': '/home/ksen/PycharmProjects/movie-screenplays/data/movie_screenplay_dataset/movie_characters/labels',
            'imdb_chars_train_data': '/home/ksen/PycharmProjects/movie-screenplays/data/movie_screenplay_dataset/movie_characters/tokenized/imdb_chars_train_data.pickle',
            'logs_dir': '/home/ksen/PycharmProjects/movie-screenplays/ml_analysis/logs',
            'mlruns_dir': './mlruns/',
            'ckpt_dir': '/home/ksen/PycharmProjects/movie-screenplays/ml_analysis/logs/ckpts/'
        },
    'tokenization':
        {
            'screenplay_dataset':
                {
                    'truncation': True,
                    'padding': 'max_length',
                    'max_scene_number': 10
                },
            'character_dataset':
                {
                    'max_seq_length': 512
                },
            'next_scene_dataset':
                {
                    'max_nrof_pairs': 1000,
                    'max_seq_length': 512,
                    'truncation': True,
                    'padding': 'max_length'
                },
            'nrof_scene_symbols':
                {
                    'min': 50,
                    'max': 20000
                }
        },
    'train':
        {
            'task': 'movie_script_awards',
            'exp_name': '100_scenes_average',
            'to_validate': False,
            'validation_step': 20,
            'batch_size':
                {
                    'train': 4,
                    'val': 4,
                    'test': 4
                },
            'test_val_split': 0.2,
            'dropout': 0.1,
            'optimizer': 'AdamW',
            'optimizer_params':
                {
                    'AdamW':
                        {
                            'lr': 1e-5,
                            'eps': 1e-8,
                            'weight_decay': 0.0001
                        }
                },
            'embedding_size': 768,
            'model': BertModel,  # from [BertForSequenceClassification, LongformerForSequenceClassification]
            'tokenizer': BertTokenizerFast,  # from [BertTokenizer, LongformerTokenizer]
            'pretrained_model_type': 'bert-base-cased',  # from ['bert-base-cased', 'allenai/longformer-base-4096']
            'nrof_steps_for_shed': 800,
            'nrof_warmup_steps': 50,
            'train_logging_step': 10,
            'metric_eval_examples_num': 50,
            'nrof_epochs': 2,
            'task_name': 'script_awards',
            'classes_names':
                {
                    'movie_genre':
                        [
                            'Comedy', 'Short', 'Thriller', 'Documentary', 'Horror', 'Biography', 'Mystery', 'Animation',
                            'Crime', 'Adventure', 'Action', 'Drama', 'Fantasy'
                        ],
                    'movie_script_awards':
                        [
                            'Not nominated', 'Nominated'
                        ],
                    'character_role':
                        [
                            'Main', 'Minor'
                        ],
                    'character_gender':
                        [
                            'Female', 'Male'
                        ],
                    'next_scene_prediction':
                        [
                            'Next', 'Not next'
                        ]
                },
            'use_intermediate_weights': False,
            'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            'load_model': False,
            'global_ckpt_saving_step': 20,
            'running_ckpt_saving_step': 100,
            'seed': 11
        },
    'screenplay_elements': Enum('screenplay_elements', 'scene_heading speaker_heading dialog text'),
    'tasks': Enum('tasks', 'character_role character_gender movie_script_awards movie_genre next_scene_prediction')
}
