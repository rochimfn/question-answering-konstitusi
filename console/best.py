import collections
import logging
import multiprocessing as mp
from pathlib import Path

import pandas as pd
from gensim.utils import simple_preprocess

from rc_modules import Doc2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
logging.getLogger().setLevel(logging.CRITICAL)


def load_data():
    cache = Path('.cache/dataframe.pickle')
    if cache.is_file():
        return pd.read_pickle(cache)
    else:
        url = 'https://raw.githubusercontent.com/rochimfn/tanyahukum-bot/main/dataset.csv'
        df = pd.read_csv(url, sep='\t', quotechar='\'')
        df['dataset'] = df['Context'] + ' ' + df['Response']
        df.to_pickle('.cache/dataframe')
        return df


def f_vector_size(vector_size):
    df = load_data()
    options = {
        'min_count': 2,
        'vector_size': vector_size,
        'epochs': 1
    }
    doc2vec = Doc2vec()
    doc2vec.set_dataframe(dataframe=df, dataset_key='dataset')
    doc2vec.set_options(options)
    ranks = get_ranks(doc2vec, df)
    counter = dict(sorted(collections.Counter(ranks).items()))
    print(f'{vector_size} vector')
    counter['vector_size'] = vector_size
    return counter


def f_epochs(epochs):
    df = load_data()
    options = {
        'min_count': 2,
        'vector_size': 20,
        'epochs': epochs
    }
    doc2vec = Doc2vec(df, 'dataset', options)
    ranks = get_ranks(doc2vec, df)
    counter = dict(sorted(collections.Counter(ranks).items()))
    print(f'{epochs} epochs')
    counter['epochs'] = epochs
    return counter


def get_ranks(model, dataframe):
    for index, row in dataframe.iterrows():
        tokens = simple_preprocess(row['Context'])
        results = model.ask(tokens)
        try:
            yield results.index.to_list().index(index)
        except ValueError:
            yield -1


def main():
    with mp.Pool(6) as pool:
        result = pool.map(f_epochs, range(1, 101))
        pd.DataFrame(result).to_csv('result_based_epochs.csv', index=False)

        result = pool.map(f_vector_size, range(1, 101))
        pd.DataFrame(result).to_csv('result_based_vector_size.csv', index=False)


if __name__ == '__main__':
    main()
