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

    models = {'tfidf': Tfidf(), 'doc2vec': Doc2vec(), 'word2vec': Word2vec()}
    for name, model in models.items():
        model.set_dataframe(dataframe=df, dataset_key='dataset')
        model.create_cache(f'.cache/{name}')


if __name__ == '__main__':
    main()
