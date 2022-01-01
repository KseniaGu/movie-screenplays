import re


def get_year(movie_name, min_year=1900, max_year=2022):
    matches = re.findall(r'\d{4}', movie_name)[::-1]
    matches = filter(lambda m: int(m) > min_year and int(m) < max_year, matches)
    return next(iter(matches), None)


def get_year_from(text_to_search):
    date = re.findall(r'\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}', text_to_search)
    date = date[-1] if date else get_year(text_to_search)

    if date:
        if date[-4:].isdecimal():
            return int(date[-4:])
        elif date[-2:].isdecimal():
            if int(date[-2]) > 2:
                return int('19' + date[-2:])
            else:
                return int('20' + date[-2:])

    return None


def find_text_to_search(script_text, anno_dict):
    script_text = re.sub('(\n)+', '\n', script_text)
    script_text = ' '.join(script_text.split('\n'))
    script_text = re.sub(' +', ' ', script_text)
    text_to_search = script_text
    nrof_attempts = 0

    while len(text_to_search) > len(script_text) // 2 and nrof_attempts < 20:
        first_text = ''

        for i, scene in enumerate(anno_dict):
            for j, segment in enumerate(scene):
                ht_items = segment['head_text'].items()
                for key, val in ht_items:
                    if val and not isinstance(val, list):
                        first_text = val
                        nrof_attempts += 1
                        anno_dict[i][j]['head_text'][key] = ''
                        break
                if first_text:
                    break
            if first_text:
                break

        text_to_search = script_text[:script_text.find(first_text)]
    return text_to_search