import os
import requests
import praw
from bs4 import BeautifulSoup

from data.parsing.web_database.movie_meta.titles import TitleNames
from data.parsing.web_database.wd_utils import save_as_is


class ScreenplayScraper:
    """Screenplay data scraping. Not all sources added."""

    def __init__(self, config):
        self.title_names = TitleNames(config)
        self.config = config
        self.union = self._get_scripts_union()

    def _get_scripts_union(self):
        union = set()

        for script_name in os.listdir(os.path.join(self.config.scripts_dir)):
            union.update((script_name[:script_name.find('_')],))

        return union

    def scrape(self, source_name):

        if source_name == self.config.sources_names.scriptorama.name:
            for table_link in self.config.scriptorama['table_links']:
                website_url = requests.get(table_link, timeout=5).text
                soup = BeautifulSoup(website_url, "html.parser")
                start_index, stop_index = 3, -7
                all_p = soup.find_all('p')

                for i, p in enumerate(all_p[start_index:stop_index]):
                    title_name = self.title_names.preprocess(source_name, p.contents[0].text)
                    try:
                        if title_name in self.union or title_name is None:
                            continue
                        try:
                            link = p.a['href']
                            response = requests.get(link)
                            ext = save_as_is(link.split('.')[-1])

                            if ext is not None:
                                with open(os.path.join(self.config.scripts_dir, title_name + "_script." + ext),
                                          'wb') as f:
                                    f.write(response.content)
                                print('File written')
                            else:
                                title_soup = BeautifulSoup(response.text, "html.parser")
                                script = title_soup.get_text()
                                with open(os.path.join(self.config.scripts_dir, title_name + "_script.txt"), 'w',
                                          encoding='utf-8') as f:
                                    f.write(script)
                                print('File written')

                        except Exception as e:
                            print(e)
                    except Exception as e:
                        print(f'Something went wrong. Title name: {title_name}\nException:\n{e}')
        elif source_name == self.config.sources_names.simplyscripts.name:
            website_url = open(self.config.simplyscripts['page_path'], 'r', encoding='utf-8')
            soup = BeautifulSoup(website_url, "html.parser")
            all_p = soup.find_all('p')
            print(all_p[-1])

            for i, p in enumerate(all_p):
                title_name = self.title_names.preprocess(source_name, p.contents[0].text)

                if title_name in self.union or title_name is None:
                    continue
                try:
                    link = p.a['href']
                    response = requests.get(link)
                    ext = save_as_is(link.split('.')[-1])

                    if ext is not None:
                        with open(os.path.join(self.config.scripts_dir, title_name + "_script." + ext), 'wb') as f:
                            f.write(response.content)
                    else:
                        title_soup = BeautifulSoup(response.text, "html.parser")
                        script = title_soup.get_text()
                        with open(os.path.join('simply_scripts', title_name + "_script.txt"), 'w',
                                  encoding='utf-8') as f:
                            f.write(script)

                    print('File written')
                except Exception as e:
                    print(f'Something went wrong. Title name: {title_name}\nException:\n{e}')
        elif source_name == self.config.sources_names.gotothehistory.name:
            website_url = requests.get(self.config.gotothehistory['page_path'], timeout=5).text
            soup = BeautifulSoup(website_url, "html.parser")
            all_p = soup.find_all('p')
            soup_p = BeautifulSoup(str(all_p[3]), "html.parser")

            for i, a in enumerate(soup_p.find_all('a')):
                title_name = self.title_names.preprocess(source_name, a.contents[0])

                if title_name in self.union or title_name is None:
                    continue
                try:
                    link = a['href']
                    response = requests.get(link)
                    ext = save_as_is(link.split('.')[-1])

                    if ext is not None:
                        with open(os.path.join(self.config.scripts_dir, title_name + "_script." + ext), 'wb') as f:
                            f.write(response.content)
                    else:
                        title_soup = BeautifulSoup(response.text, "html.parser")
                        script = title_soup.get_text()
                        with open(os.path.join(self.config.scripts_dir, title_name + "_script.txt"), 'w',
                                  encoding='utf-8') as f:
                            f.write(script)
                    print('File written')

                except Exception as e:
                    print(f'Something went wrong. Title name: {title_name}\nException:\n{e}')

        elif source_name == self.config.sources_names.dailyscript.name:
            website_url = open(self.config.dailyscript['page_path'], 'r', encoding='utf-8')
            soup = BeautifulSoup(website_url, "html.parser")
            all_p = soup.find_all('p')

            for i, p in enumerate(all_p):
                title_name = self.title_names.preprocess(source_name, p.a.contents[0])

                if title_name in self.union or title_name is None:
                    continue
                try:
                    link = p.a['href']
                    response = requests.get(self.config.dailyscript['source_url'] + link)
                    ext = save_as_is(link.split('.')[-1])

                    if ext is not None:
                        with open(os.path.join('daily_scripts', title_name + "_script." + ext), 'wb') as f:
                            f.write(response.content)
                    else:
                        title_soup = BeautifulSoup(response.text, "html.parser")
                        script = title_soup.get_text()
                        with open(os.path.join('daily_scripts', title_name + "_script.txt"), 'w',
                                  encoding='utf-8') as f:
                            f.write(script)
                    print('File written')

                except Exception as e:
                    print(f'Something went wrong. Title name: {title_name}\nException:\n{e}')

        elif source_name == self.config.sources_names.awesomefilm.name:
            website_url = requests.get(self.config.awesomefilm['source_url']).text
            soup = BeautifulSoup(website_url, "html.parser")
            start, stop = 5, -4
            all_tr = soup.find_all('tr')

            for i, p in enumerate(all_tr[start:stop]):
                if not (hasattr(p, 'a') and hasattr(p.a, 'contents')):
                    continue
                title_name = self.title_names.preprocess(p.a.contents[0])

                if title_name in self.union or title_name is None:
                    continue
                try:
                    link = p.a['href']
                    response = requests.get(self.config.awesomefilm['source_url'] + link)
                    ext = save_as_is(link.split('.')[-1])

                    if ext is not None:
                        with open(os.path.join('awesome_scripts', title_name + "_script." + ext), 'wb') as f:
                            f.write(response.content)
                    else:
                        title_soup = BeautifulSoup(response.text, "html.parser")
                        script = title_soup.get_text()
                        with open(os.path.join('awesome_scripts', title_name + "_script.txt"), 'w',
                                  encoding='utf-8') as f:
                            f.write(script)
                    print('File written')

                except Exception as e:
                    print(f'Something went wrong. Title name: {title_name}\nException:\n{e}')
        elif source_name == self.config.sources_names.reddit.name:
            """Identification data is hidden."""
            reddit = praw.Reddit(client_id="...",
                                 client_secret="...",
                                 password="...",
                                 user_agent="...",
                                 username="...")
            subreddit = reddit.subreddit("Screenwriting")
            top_posts = subreddit.new(limit=500)

            for i, post in enumerate(top_posts):
                try:
                    flair = post.link_flair_text
                    if flair == 'SCRIPT REQUEST':
                        print(i, ' ', post.title)
                        comments = post.comments
                        for comment in comments:
                            print(comment.body)
                        print()
                except:
                    continue
        ...
