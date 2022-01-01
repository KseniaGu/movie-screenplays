import re


def remove_brackets(name):
    pattern = r"\([a-z0-9 ]+\)"
    return re.sub(pattern, '', name)


def remove_non_alpha_numeric(name):
    pattern = '[^A-Za-z0-9 ]+'
    return re.sub(pattern, '', name)


def remove_extra_spaces(name):
    return ' '.join(name.split())


def save_as_is(extension):
    if extension in ('doc', 'txt', 'rtf', 'RTF', 'pdf', 'PDF', 'TXT'):
        return extension
    return None
