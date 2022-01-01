import re
from data.parsing.web_database.wd_utils import remove_non_alpha_numeric, remove_extra_spaces


class TitleNames:
    """Processing title names of downloaded screenplays."""

    def __init__(self, config):
        self.config = config

    @staticmethod
    def base_preprocess(title):
        title = remove_extra_spaces(title)
        title = remove_non_alpha_numeric(title.lower())
        title = title.strip()
        return title

    def preprocess(self, source_name, title, for_scraping=False):
        if source_name == self.config.sources_names.imsdb.name:
            if not title.startswith('Written') and title != '\n':
                if not title.find('(') == -1:
                    title = title[:title.find('(')]
                title = title.strip()
                if not for_scraping:
                    title = remove_non_alpha_numeric(title.lower())
                return title

        elif source_name == self.config.sources_names.scriptorama.name:
            pattern = r'\b\w+\b'
            title = re.findall(pattern, title)
            title = ' '.join(title)
            return remove_non_alpha_numeric(title.lower())

        elif source_name == self.config.sources_names.simplyscripts.name:
            if not title.find('(') == -1:
                title = title[:title.find('(')]
            return self.base_preprocess(title)

        elif source_name == self.config.sources_names.gotothehistory.name:
            return self.base_preprocess(title)

        elif source_name == self.config.sources_names.dailyscript.name:
            if not 'filmed as' in title:
                if title.find('(') != -1:
                    title = title[:title.find('(')]
                title = remove_non_alpha_numeric(title.lower())
                if not title.find('by') == -1:
                    title = title[:title.find('by')]
                title = title.strip()
                return title
            else:
                return None

        elif source_name == self.config.sources_names.awesomefilm.name:
            return self.base_preprocess(title)

        elif source_name == self.config.sources_names.slug.name:
            title = self.base_preprocess(title)
            if not title.find('script pdf download') == -1:
                title = title[:title.find('script pdf download')]
            title = title.strip()
            return title

        elif source_name == self.config.sources_names.horror.name:
            return self.base_preprocess(title)

        elif source_name == self.config.sources_names.scriptologist.name:
            return self.base_preprocess(title)

        elif source_name == self.config.sources_names.sfy.name:
            if not 'filmed as' in title:
                title = remove_extra_spaces(title)
                title = remove_non_alpha_numeric(title.lower())
                if not title.find('(') == -1:
                    title = title[:title.find('(')]
                title = re.sub(r'[0-9]{4,}', '', title)
                title = title.strip()
                return title
            else:
                return None

        elif source_name == self.config.sources_names.scifi.name:
            return self.base_preprocess(title)

        elif source_name == self.config.sources_names.dailyactor.name:
            return self.base_preprocess(title)

        elif source_name == self.config.sources_names.screenplays_online.name:
            return self.base_preprocess(title)

        elif source_name == self.config.sources_names.thescriptlab.name:
            return self.base_preprocess(title)

        elif source_name == self.config.sources_names.onscreen.name:
            title = remove_extra_spaces(title)
            title = remove_non_alpha_numeric(title.lower())
            title = title.replace('script', '')
            title = title.strip()
            return title

        elif source_name == self.config.sources_names.selling.name:
            title = title.lower()
            if 'filmed as' in title:
                return None
            if not title.find('(') == -1:
                title = title[:title.find('(')]
            return self.base_preprocess(title)

        return None
