import re
import os
import json

from utils.common import read_file

"""
Getting characters and their texts from annotations made with ScreenPy (https://github.com/drwiner/ScreenPy)
"""


def get_anno_characters(anno):
    speaker_title_characters, subject_characters = set(), set()

    for j, scene in enumerate(anno):
        for i, segment in enumerate(scene):
            segment_dict = segment['head_text']
            if segment['head_type'] == 'heading':
                if 'subj' in segment_dict and segment_dict['subj']:
                    subject_characters.add(segment_dict['subj'])

            if segment['head_type'] == "speaker/title":
                if "speaker/title" in segment_dict and segment_dict["speaker/title"]:
                    speaker_title_characters.add(segment_dict["speaker/title"])

    return list(speaker_title_characters), list(subject_characters)


def get_all_anno_characters(annotations_dir):
    all_anno_chars = {}

    for i, anno_file in enumerate(os.listdir(annotations_dir)):
        imdb_id = anno_file[anno_file.find('_') + 1:anno_file.find('.')]
        anno = read_file(os.path.join(annotations_dir, anno_file))
        chars_speaker_list, chars_subj_list = get_anno_characters(anno)
        all_anno_chars[imdb_id] = chars_speaker_list if chars_speaker_list else chars_subj_list

    return all_anno_chars


def process_name(char_name):
    return re.sub('\([a-zA-Z0-9 \-,./_\'\":]+\)', '', char_name)


def get_characters_text_from_annotation(imdb_id_to_character_matches, annotations_dir):
    """
    Ges character texts: dialogs and description of scenes where character is engaged.
    """
    imdb_id_to_characters_to_text = {}

    for i, anno_file in enumerate(os.listdir(annotations_dir)):
        imdb_id = anno_file[anno_file.find('_') + 1:anno_file.find('.')]
        if not imdb_id in imdb_id_to_character_matches:
            continue
        anno_to_imdb_characters = [(x, [imdb_char for imdb_char, _, _ in imdb_chars_with_scores]) \
                                   for x, imdb_chars_with_scores in
                                   imdb_id_to_character_matches[imdb_id].items() \
                                   if imdb_chars_with_scores]
        imdb_id_to_characters_to_text[imdb_id] = {}
        for _, imdb_chars in anno_to_imdb_characters:
            for imdb_char in imdb_chars:
                imdb_id_to_characters_to_text[imdb_id][imdb_char] = []

        with open(os.path.join(annotations_dir, anno_file)) as f:
            anno = json.load(f)

        for j, scene in enumerate(anno):
            for i, segment in enumerate(scene):
                if not 'text' in segment:
                    continue
                segment_text = segment['text']
                for chars in list(anno_to_imdb_characters):
                    if segment['head_type'] == 'heading':
                        if not 'subj' in segment['head_text'] or segment['head_text']['subj'] is None:
                            continue
                        if segment['head_text']['subj'] == chars[0]:
                            segment_text_to_add = str(j) + ') ' + str(i) + ') dialog: ' + segment_text
                            for imdb_char in chars[1]:
                                if not segment_text_to_add in imdb_id_to_characters_to_text[imdb_id][imdb_char]:
                                    imdb_id_to_characters_to_text[imdb_id][imdb_char].append(segment_text_to_add)
                        else:
                            low_segment = segment_text.lower()
                            segment_text_to_add = str(j) + ') ' + str(i) + ') text: ' + segment_text
                            for imdb_char in chars[1]:
                                if (chars[0].lower() in low_segment
                                    or process_name(chars[0]).lower() in low_segment
                                    or imdb_char.lower() in low_segment) \
                                        and not segment_text_to_add in imdb_id_to_characters_to_text[imdb_id][imdb_char]:
                                    imdb_id_to_characters_to_text[imdb_id][imdb_char].append(segment_text_to_add)
                    elif segment['head_type'] == 'speaker/title':
                        if not 'speaker/title' in segment['head_text'] or segment['head_text']['speaker/title'] != \
                                chars[0]:
                            continue
                        segment_text_to_add = str(j) + ') ' + str(i) + ') dialog: ' + segment_text
                        for imdb_char in chars[1]:
                            if not segment_text_to_add in imdb_id_to_characters_to_text[imdb_id][imdb_char]:
                                imdb_id_to_characters_to_text[imdb_id][imdb_char].append(segment_text_to_add)

    return imdb_id_to_characters_to_text
