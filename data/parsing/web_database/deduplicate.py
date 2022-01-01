import os


def find_id_by_year(movie_year, imdb_list):
    """Chooses imdb id for movie year from all imdb movies with matching name by nearest year"""
    new_value, min_diff = [], 10000

    for title, rat, vot, year, atr_list in imdb_list:
        if not year:
            continue
        diff = abs(movie_year - year)

        if not diff <= min_diff:
            continue

        if not new_value:
            new_value = [title, rat, vot, year, atr_list]
        else:
            if diff < min_diff:
                new_value = [title, rat, vot, year, atr_list]
            else:
                if vot and new_value[2]:
                    if vot > new_value[2]:
                        new_value = [title, rat, vot, year, atr_list]
                elif not new_value[2]:
                    new_value = [title, rat, vot, year, atr_list]
        min_diff = diff

    return new_value


def get_all_movies_imdb_id(scripts_dir, movies_meta_data_dict, movie_years_dict):
    not_checked_movies, all_movies_imdb_titles = [], {}

    for x in os.listdir(os.path.join(scripts_dir)):
        if not (x.endswith('_scripts') and os.path.isdir(os.path.join(scripts_dir, x))):
            continue
        for script_name in os.listdir(os.path.join(scripts_dir, x)):
            if not script_name.endswith('.txt'):
                continue
            movie_name = script_name[:script_name.find('_')]

            if movie_name in movies_meta_data_dict:
                all_movies_imdb_titles[movie_name] = []

                if movie_name in movie_years_dict:
                    for year in movie_years_dict[movie_name]:
                        dict_value = find_id_by_year(year, movies_meta_data_dict[movie_name])
                        if dict_value:
                            all_movies_imdb_titles[movie_name].append(dict_value)
                else:
                    all_movies_imdb_titles[movie_name].append(movies_meta_data_dict[movie_name][0])
            else:
                print('not found movie:', movie_name)
                not_checked_movies.append(movie_name)

    return all_movies_imdb_titles, not_checked_movies
