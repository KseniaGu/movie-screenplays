import re


class BasePreprocessor:
    @staticmethod
    def preprocess_base(text):
        text = re.sub(" +|\n+|\r|\t|\0|\x0b|\xa0|\x80|\x93|\x99", ' ', text)
        text = ' '.join(re.findall(r'[a-zA-Z0-9!"#%&\'()+,\-\./:;?\[\]]+', text))
        text = re.sub(' \.', '../preprocessing', text)
        return text.strip()

