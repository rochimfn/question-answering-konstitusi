import logging

import pandas as pd

from rc_modules import Word2vec, Doc2vec, Tfidf

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
logging.getLogger().setLevel(logging.CRITICAL)


def load_data():
    url = 'https://raw.githubusercontent.com/rochimfn/tanyahukum-bot/main/dataset.csv'
    df = pd.read_csv(url, sep='\t', quotechar='\'')
    df['dataset'] = df['Context'] + ' ' + df['Response']
    return df


def main():
    models = (Tfidf(), Word2vec(), Doc2vec())
    df = load_data()
    for model in models:
        model.set_dataframe(dataframe=df, dataset_key='dataset')
        counter = 0
        for index, row in df.iterrows():
            question = row['Context']
            answer = model.ask(question)

            if index in answer.index.to_list() and answer.index.to_list().index(index) == 0:
                counter += 1

        print(f'{type(model).__name__} : {counter}')


if __name__ == '__main__':
    main()
