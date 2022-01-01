import os
from collections import OrderedDict
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz

from data.parsing.movie_characters.config import MovieCharactersConfig as chars_cfg
from utils.common import read_file

stopWords = set(stopwords.words('english'))

"""
Matching characters found from annotations to characters from IMDB.com
"""


def get_tokens_for(imdb_id, character, character_source='anno'):
    imdbs_id_to_anno_chars_to_tokens = read_file(chars_cfg.imdbs_id_to_anno_chars_to_tokens_path)
    imdbs_id_to_imdb_chars_to_tokens = read_file(chars_cfg.imdbs_id_to_imdb_chars_to_tokens_path)

    if character_source == 'anno':
        if not imdb_id in imdbs_id_to_anno_chars_to_tokens \
                or not character in imdbs_id_to_anno_chars_to_tokens[imdb_id]:
            return None
        tokens, depprels, pos = imdbs_id_to_anno_chars_to_tokens[imdb_id][character]
    else:
        if not imdb_id in imdbs_id_to_imdb_chars_to_tokens \
                or not character in imdbs_id_to_imdb_chars_to_tokens[imdb_id]:
            return None
        tokens, depprels, pos = imdbs_id_to_imdb_chars_to_tokens[imdb_id][character]

    deprels = [x for i, x in enumerate(depprels) if
               not (pos[i] == 'PUNCT' or pos[i] == 'SYM' or x == 'punct') or tokens[i] == '/']
    tokens = [x.lower() for i, x in enumerate(tokens) if
              not (pos[i] == 'PUNCT' or pos[i] == 'SYM' or depprels[i] == 'punct') or x == '/']

    return tokens, deprels


def get_iou_score(anno_tokens, imdb_tokens, anno_deprels, imdb_deprels, full_subseq):
    """
    Gets IOU score for tokens for characters from IMDB and characters from annotations.

    Args:
        anno_tokens (list): tokens for character from annotations
        imdb_tokens (list): tokens for character from IMDB
        anno_deprels (list): dependency relation for tokens for character from annotations
        imdb_deprels (list): dependency relation for tokens for character from IMDB
        full_subseq (bool): if one of the characters is full subsequence of another
    Returns: IOU score for crossing tokens
    """
    imdb_tokens_set, anno_tokens_set = set(imdb_tokens), set(anno_tokens)
    crossing = imdb_tokens_set & anno_tokens_set

    if not full_subseq:
        crossing = crossing - stopWords - set('.') - chars_cfg.dataset_stop_words - set('/')
    else:
        crossing = crossing - stopWords - set('.') - set('/')

    if crossing:
        if not any([imdb_deprels[imdb_tokens.index(cross)] in ('flat', 'root', 'nsubj', 'conj') \
                    and anno_deprels[anno_tokens.index(cross)] in ('flat', 'root', 'nsubj', 'conj')
                    for cross in crossing]):
            crossing = set()
        else:
            left_anno_tokens = list(anno_tokens_set - stopWords - set('.') - crossing)
            left_imdb_tokens = list(imdb_tokens_set - stopWords - set('.') - crossing)

            if len(left_anno_tokens) == 1 and len(left_imdb_tokens) == 1 \
                    and left_anno_tokens[0].isdecimal() and left_imdb_tokens[0].isdecimal() \
                    and int(left_anno_tokens[0]) != int(left_imdb_tokens[0]):
                crossing = set()

    crossing = len([x for x in crossing if len(x) > 1])
    imdb_tokens = [x for x in imdb_tokens if len(x) > 1]
    anno_tokens = [x for x in anno_tokens if len(x) > 1]
    crossing /= len(set(imdb_tokens) | set(anno_tokens) - stopWords - set('.'))

    return crossing


def get_ld_score(anno_tokens, imdb_tokens):
    """Gets Levenshtein distance between character names."""
    return fuzz.ratio(' '.join(anno_tokens), ' '.join(imdb_tokens))


def get_full_subseq(anno_tokens, imdb_tokens):
    joined_anno = ' '.join(anno_tokens)
    joined_imdb = ' '.join(imdb_tokens)

    if '/' in joined_anno and len(anno_tokens) <= 4:
        return anno_tokens[max(0, anno_tokens.index('/') - 1)] in imdb_tokens or anno_tokens[
            min(len(anno_tokens) - 1, anno_tokens.index('/') + 1)] in imdb_tokens
    if ' and ' in joined_anno and len(anno_tokens) <= 4:
        return anno_tokens[max(0, anno_tokens.index('and') - 1)] in imdb_tokens or anno_tokens[
            min(len(anno_tokens) - 1, anno_tokens.index('and') + 1)] in imdb_tokens

    return joined_anno in joined_imdb or joined_imdb in joined_anno


def check_for_crossing(anno_characters, imdb_characters, imdb_id):
    """Finds all IMDB character matches for characters from annotations."""
    imdbs_id_to_anno_chars_to_tokens = read_file(chars_cfg.imdbs_id_to_anno_chars_to_tokens_path)

    if not imdb_characters:
        return dict([(character, None) for character in anno_characters])

    imdb_characters = list(OrderedDict.fromkeys(imdb_characters))
    anno_characters_matches = dict(zip(anno_characters, [[(None, 0.)] for _ in range(len(anno_characters))]))

    for i, anno_character in enumerate(anno_characters):
        if imdb_id in imdbs_id_to_anno_chars_to_tokens and anno_character in imdbs_id_to_anno_chars_to_tokens[imdb_id]:
            anno_tokens = get_tokens_for(imdb_id, anno_character, character_source='anno')
            if anno_tokens:
                anno_tokens, anno_deprels = anno_tokens
                for j, imdb_character in enumerate(imdb_characters):
                    imdb_tokens = get_tokens_for(imdb_id, imdb_character, character_source='imdb')
                    if imdb_tokens:
                        imdb_tokens, imdb_deprels = imdb_tokens
                        if imdb_tokens and anno_tokens and float(len(anno_tokens)) / len(
                                imdb_tokens) <= chars_cfg.anno_imdb_chars_ratio:
                            ld_crossing = get_ld_score(anno_tokens, imdb_tokens)
                            full_subseq = get_full_subseq(anno_tokens, imdb_tokens)
                            crossing = get_iou_score(anno_tokens, imdb_tokens, anno_deprels, imdb_deprels, full_subseq)

                            if crossing > 0:
                                anno_characters_matches[anno_characters[i]].append(
                                    (imdb_characters[j], crossing, ld_crossing, full_subseq))

    return anno_characters_matches


def get_characters_matches(imdb_charactrers_dict, anno_characters_dict):
    imdb_id_to_character_matches = {}

    for i, anno_file in enumerate(os.listdir(chars_cfg.annotations_path)):
        imdb_id = anno_file[anno_file.find('_') + 1:anno_file.find('.')]
        if not imdb_id in imdb_id_to_character_matches:
            imdb_characters = imdb_charactrers_dict[imdb_id]
            anno_characters = anno_characters_dict[imdb_id]
            imdb_id_to_character_matches[imdb_id] = check_for_crossing(anno_characters, imdb_characters, imdb_id)

    return imdb_id_to_character_matches
