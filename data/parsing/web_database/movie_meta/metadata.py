import re
import requests
from bs4 import BeautifulSoup

from utils.common import write_file


def full_meta(full, imdb_to_meta, full_meta_keys, to_save=False):
    for i, (key, item) in enumerate(full.items()):
        imdb_key = item['imdbID']

        if not isinstance(imdb_key, str):
            imdb_key = str(imdb_key)

        if not imdb_key in imdb_to_meta:
            imdb_to_meta[imdb_key] = {}

        for item_key in full_meta_keys:
            if isinstance(item[item_key], str):
                if item[item_key]:
                    imdb_to_meta[imdb_key][item_key] = item[item_key]
                else:
                    imdb_to_meta[imdb_key][item_key] = ''
            elif isinstance(item[item_key], list):
                text = ''.join([val + ', ' for val in item[item_key]])
                if len(text) > 2:
                    text = text[:-2]
                imdb_to_meta[imdb_key][item_key] = text
            elif isinstance(item[item_key], dict):
                for dict_key, value in item[item_key].items():
                    if value:
                        imdb_to_meta[imdb_key][dict_key] = value
                    else:
                        imdb_to_meta[imdb_key][dict_key] = ''
            elif item[item_key] is None:
                imdb_to_meta[imdb_key][dict_key] = ''

    if to_save:
        write_file(imdb_to_meta, 'all_meta_data.pickle')

    return imdb_to_meta


def rat_vot_meta(imdb_to_meta, rat_vot, to_save=False):
    for i, (key, item) in enumerate(rat_vot.items()):
        imdb_key = key

        if not isinstance(imdb_key, str):
            imdb_key = str(imdb_key)

        if not imdb_key in imdb_to_meta:
            imdb_to_meta[imdb_key] = {}
        if item[0]:
            imdb_to_meta[imdb_key]['imdb user rating'] = str(item[0])
        else:
            imdb_to_meta[imdb_key]['imdb user rating'] = ''
        if item[1]:
            imdb_to_meta[imdb_key]['number of imdb user votes'] = str(item[1])
        else:
            imdb_to_meta[imdb_key]['number of imdb user votes'] = ''

    if to_save:
        write_file(imdb_to_meta, 'all_meta_data.pickle')

    return imdb_to_meta


def years_meta(imdb_to_meta, years, to_save=False):
    for i, (key, item) in enumerate(years.items()):
        imdb_key = key
        if not isinstance(imdb_key, str):
            imdb_key = str(imdb_key)
        if item:
            imdb_to_meta[imdb_key]['year'] = str(item)
        else:
            imdb_to_meta[imdb_key]['year'] = ''

    if to_save:
        write_file(imdb_to_meta, 'all_meta_data.pickle')

    return imdb_to_meta


def other_meta(imdb_to_meta, last, last_meta_scores, to_save=False):
    for i, (key, item) in enumerate(last.items()):
        imdb_key = key

        if not isinstance(imdb_key, str):
            imdb_key = str(imdb_key)
        for item_key in last_meta_scores:
            if isinstance(item[item_key], str):
                imdb_to_meta[imdb_key][item_key] = item[item_key]

            if isinstance(item[item_key], list) and not item_key == 'keywords':
                imdb_to_meta[imdb_key][item_key] = ''

                for it in item[item_key]:
                    if isinstance(it, str):
                        imdb_to_meta[imdb_key][item_key] += it + ', '
                    if isinstance(it, dict):
                        award = it['award']
                        award_year = str(it['year'])
                        if not award + ' ' + award_year + ', ' in imdb_to_meta[imdb_key][item_key]:
                            imdb_to_meta[imdb_key][item_key] += award + ' ' + award_year + ', '

                if ', ' in imdb_to_meta[imdb_key][item_key]:
                    imdb_to_meta[imdb_key][item_key] = imdb_to_meta[imdb_key][item_key][:-2]

            if isinstance(item[item_key], tuple) and not item_key == 'keywords':
                if item[item_key][0]:
                    text = ', '.join([val for val in item[item_key][0]])
                else:
                    text = ''

                if item[item_key][1]:
                    text += str(item[item_key][1])

                imdb_to_meta[imdb_key][item_key] = text
                if ', ' in imdb_to_meta[imdb_key][item_key]:
                    imdb_to_meta[imdb_key][item_key] = imdb_to_meta[imdb_key][item_key][:-2]
            elif (isinstance(item[item_key], tuple) or isinstance(item[item_key], list)) and item_key == 'keywords':
                imdb_to_meta[imdb_key][item_key] = ''

                for it in item[item_key]:
                    imdb_to_meta[imdb_key][item_key] += it + ', '

                if ', ' in imdb_to_meta[imdb_key][item_key]:
                    imdb_to_meta[imdb_key][item_key] = imdb_to_meta[imdb_key][item_key][:-2]

    if to_save:
        write_file(imdb_to_meta, 'all_meta_data.pickle')

    return imdb_to_meta


def gather_all_meta(full, rat_vot, years, last, full_meta_keys, last_meta_scores):
    imdb_to_meta = {}

    ## full meta
    imdb_to_meta = full_meta(full, imdb_to_meta, full_meta_keys)

    ## for rat vot meta
    imdb_to_meta = rat_vot_meta(imdb_to_meta, rat_vot)

    ## for years meta
    imdb_to_meta = years_meta(imdb_to_meta, years)

    ## for other meta
    imdb_to_meta = other_meta(imdb_to_meta, last, last_meta_scores)

    return imdb_to_meta


def meta_critic_scrape(link):
    headers = {'User-Agent': 'Mozilla/5.0'}
    website_url = requests.get(link, headers=headers).text
    soup = BeautifulSoup(website_url, "html.parser")
    all_div = soup.find_all('div')
    scores, reviews = [], []

    for a in all_div:
        try:
            bools = [hasattr(a.contents[i], 'class') for i in range(len(a.contents))]
            for i, class_ in enumerate(bools):
                if not class_:
                    continue
                if 'metascore_w' in a.contents[i]['class']:
                    scores.append(a.contents[i].text)
                if a.contents[i]['class'] == ['summary']:
                    reviews.append(a.contents[i].text)
        except Exception:
            pass

    reviews = [review.replace('\n', '') for review in reviews]
    reviews = [re.sub(' +', ' ', review) for review in reviews]
    score_review_dict = dict(zip(scores, reviews))

    return score_review_dict
