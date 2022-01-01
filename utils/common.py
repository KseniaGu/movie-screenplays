import json
import os
import pickle
import numpy as np
import pandas as pd


def not_na(x):
    return x and x == x


def write_file(file, path):
    extension = os.path.splitext(path)[1]
    if extension == '.pickle':
        with open(path, 'wb') as f:
            pickle.dump(file, f)
    elif extension == '.csv':
        file.to_csv(path, index=False)
    elif extension == '.xlsx':
        file.to_excel(path, index=False)
    elif extension == '.npy':
        np.save(path, file)


def read_file(path):
    extension = os.path.splitext(path)[1]
    try:
        if extension == '.pickle':
            return pickle.load(open(path, 'rb'))
        elif extension == '.csv':
            return pd.read_csv(path)
        elif extension == '.xlsx':
            return pd.read_excel(path)
        elif extension == '.npy':
            return np.load(path)
        elif extension == '.txt':
            return open(path, 'r').read()
        elif extension == '.json':
            return json.load(open(path))
        else:
            print('Unknown extension')
            return None
    except FileNotFoundError:
        print('File not found')
        return None
