import logging

import pandas as pd

from rc_modules import Doc2vec, Word2vec, Tfidf

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def load_data():
    url = 'https://raw.githubusercontent.com/rochimfn/tanyahukum-bot/main/dataset.csv'
    df = pd.read_csv(url, sep='\t', quotechar='\'')
    df['dataset'] = df['Context'] + ' ' + df['Response']
    return df


def main():
    df = load_data()

    tfidf = Tfidf()
    tfidf.set_dataframe(dataframe=df, dataset_key='dataset')
    tfidf.create_cache('.cache/tfidf')

    doc2vec = Doc2vec()
    doc2vec.set_dataframe(dataframe=df, dataset_key='dataset')
    doc2vec.create_cache('.cache/doc2vec')

    word2vec = Word2vec()
    word2vec.set_dataframe(dataframe=df, dataset_key='dataset')
    word2vec.create_cache('.cache/word2vec')


if __name__ == '__main__':
    main()
