import datetime
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def remove_element_names_old(self, row):
    if row.startswith(self.config['screenplay_elements'].text.name + ':'):
        row = row[len(self.config['screenplay_elements'].text.name) + 1:]
    elif row.startswith(self.config['screenplay_elements'].dialog.name + ':'):
        row = row[len(self.config['screenplay_elements'].dialog.name) + 1:]
    elif row.startswith(self.config['screenplay_elements'].scene_heading.name + ':'):
        row = row[len(self.config['screenplay_elements'].scene_heading.name) + 1:]
    elif row.startswith(self.config['screenplay_elements'].speaker_heading.name + ':'):
        row = row[len(self.config['screenplay_elements'].speaker_heading.name) + 1:]
    return row.strip()


def remove_element_names(element_names, text):
    for name in sorted(element_names, key=len, reverse=True):
        if text.startswith(name + ':'):
            text = text[len(name) + 1:]
    return text


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def plot_conf_matr(trues,
                   predicts,
                   file_path_prefix='tr_',
                   title='Confusion matrix',
                   nrof_classes=2,
                   classes_names=('not nominated', 'nominated'),
                   figsize=(7, 7)):
    results = np.zeros((nrof_classes, nrof_classes))

    for t, p in zip(trues, predicts):
        results[t][p] += 1

    df_cm = pd.DataFrame(results.astype(np.int), index=classes_names, columns=classes_names)
    plt.figure(figsize=figsize)
    ax = sns.heatmap(df_cm, annot=True, fmt='d')
    ax.set(xlabel='predicted', ylabel='actual', title=title)
    plt.savefig(file_path_prefix + 'conf_matrix.png', bbox_inches='tight')
    plt.close()
