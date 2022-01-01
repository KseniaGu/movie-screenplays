import stanza
import json
import re

stanza.download('en')


class CharNamesPreprocessor:
    def __int__(self, path_to_imdbs_id_to_anno_chars_to_tokens):
        self.nlp = stanza.Pipeline('en', processors='tokenize, pos, lemma, depparse')
        self.path_to_imdbs_id_to_anno_chars_to_tokens = path_to_imdbs_id_to_anno_chars_to_tokens

    def remove_parenthesis(self, name):
        return re.sub(',* *\([a-zA-Z0-9 \-,./_\'\":]+\)', '', name)

    def check_if_number(self, name):
        subbed = re.sub('\d+ *[.:,;\)]', '', name)
        return re.sub(' +', '', subbed) == ''

    def preprocess_anno_char(self, name):
        name = self.remove_parenthesis(name).lower()
        name = name.replace('/', ' / ')
        if self.check_if_number(name) or name == "fade out" or name == "the end" or name == "fade in":
            return None
        name = name[0].upper() + name[1:]
        if not name.endswith('.'):
            name = name + '.'
        return name

    def preprocess_imdb_char(name):
        if name == "" or 'voice' in name.lower():
            return None
        name = name.replace('/ ...', '')
        if not name.endswith('.'):
            name = name + '.'
        return name

    def get_doc_for_anno(self, text):
        processed = self.preprocess_anno_char(text)
        if not processed:
            return None
        doc = self.nlp(processed)
        return doc

    def get_doc_for_imdb(self, text):
        processed = self.preprocess_imdb_char(text)
        if not processed:
            return None
        doc = self.nlp(processed)
        return doc

    def get_tokens_for(self, imdb_to_characters, get_doc_func, save_time=250):
        imdbs_id_to_chars_to_tokens = {}

        for j, (imdb_id, characters) in enumerate(imdb_to_characters.items()):
            characters_docs = [(character, get_doc_func(character)) for character in characters]
            characters_docs = [x for x in characters_docs if x[1]]
            chars_tokens = {}

            for i, characters_doc in enumerate(characters_docs):
                tokens = list(zip(*[(word.text, word.deprel, word.upos)
                                    for sent in characters_doc[1].sentences
                                    for word in sent.words]))
                chars_tokens[characters_doc[0]] = tokens

            imdbs_id_to_chars_to_tokens[imdb_id] = chars_tokens

            if j % save_time == 0:
                with open(self.path_to_imdbs_id_to_anno_chars_to_tokens, 'w') as f:
                    json.dump(imdbs_id_to_chars_to_tokens, f)

        return imdbs_id_to_chars_to_tokens
