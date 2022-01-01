

class MovieCharactersConfig:
    dataset_stop_words = {
        'mr', 'mr.', 'mrs.', 'mrs', 'dr', 'dr.', 'lt', 'lt.', 'professor', 'doctor', 'sgt', 'sgt.', 'uncle', 'admiral',
        'miss', 'misses', 'ms', 'ms.', 'captain', 'sergeant', 'inspector', 'agent'
    }
    imdbs_id_to_anno_chars_to_tokens_path = 'data/imdbs_id_to_anno_chars_to_tokens.json'
    imdbs_id_to_imdb_chars_to_tokens_path = 'data/imdbs_id_to_imdb_chars_to_tokens.json'
    anno_imdb_chars_ratio = 3
    annotations_path = 'data/screenplay_data/data/rule_based_annotations'


