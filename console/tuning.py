import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict

import pandas as pd

from rc_modules import Word2vec, Doc2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def load_data():
    cache = Path('.cache/dataframe.pickle')
    if cache.is_file():
        return pd.read_pickle(cache)
    else:
        url = 'https://raw.githubusercontent.com/rochimfn/tanyahukum-bot/main/dataset.csv'
        df = pd.read_csv(url, sep='\t', quotechar='\'')
        df['dataset'] = df['Keywords'] + ' ' + df['Context'] + ' ' + df['Response']
        df.to_pickle('.cache/dataframe')
        return df


def count_correct(algorithm: str, options: Dict[str, int]) -> int:
    models = {
        'word2vec': Word2vec(),
        'doc2vec': Doc2vec()
    }
    df = load_data()
    model = models[algorithm]
    model.set_dataframe(dataframe=df, dataset_key='dataset')
    model.set_options(options)
    counter = 0
    for index, row in df.iterrows():
        question = row['Context']
        answer = model.ask(question)

        if index in answer.index.to_list() and answer.index.to_list().index(index) == 0:
            counter += 1
    return counter


def f_tune_vector_word2vec(vector_size: int) -> (int, int):
    logging.info(f'word2vec:vector_size:{vector_size}')
    options = {
        'size': vector_size,
        'min_count': 2,
        'iter': 40
    }
    counter = count_correct('word2vec', options)
    return vector_size, counter


def f_tune_iter_word2vec(iter_size: int) -> (int, int):
    logging.info(f'word2vec:vector_size:{iter_size}')
    options = {
        'size': 50,
        'min_count': 2,
        'iter': iter_size
    }
    counter = count_correct('word2vec', options)
    return iter_size, counter


def f_tune_vector_doc2vec(vector_size: int) -> (int, int):
    logging.info(f'doc2vec:vector_size:{vector_size}')
    options = {
        'vector_size': vector_size,
        'min_count': 2,
        'epochs': 40
    }
    counter = count_correct('doc2vec', options)
    return vector_size, counter


def f_tune_iter_doc2vec(iter_size: int) -> (int, int):
    logging.info(f'doc2vec:iter:{iter_size}')
    options = {
        'vector_size': 50,
        'min_count': 2,
        'epochs': iter_size
    }
    counter = count_correct('doc2vec', options)
    return iter_size, counter


def main():
    with mp.Pool(6) as pool:
        result = pool.map(f_tune_vector_word2vec, range(1, 101))
        pd.DataFrame(result, columns=['Vector size', 'Correct']) \
            .to_csv('.cache/tuning_vector_size_word2vec.csv', index=False)

        result = pool.map(f_tune_iter_word2vec, range(1, 101))
        pd.DataFrame(result, columns=['Iteration', 'Correct']) \
            .to_csv('.cache/tuning_iteration_word2vec.csv', index=False)

        result = pool.map(f_tune_vector_doc2vec, range(1, 101))
        pd.DataFrame(result, columns=['Vector size', 'Correct']) \
            .to_csv('.cache/tuning_vector_size_doc2vec.csv', index=False)

        result = pool.map(f_tune_iter_doc2vec, range(1, 101))
        pd.DataFrame(result, columns=['Iteration', 'Correct']) \
            .to_csv('.cache/tuning_iteration_doc2vec.csv', index=False)


if __name__ == '__main__':
    main()
