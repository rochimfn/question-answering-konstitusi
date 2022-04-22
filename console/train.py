import pandas as pd
import logging

from rc_modules import Doc2vec, Tfidf, Word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def load_data():
    url = 'https://raw.githubusercontent.com/rochimfn/tanyahukum-bot/main/dataset.csv'
    df = pd.read_csv(url, sep='\t', quotechar='\'')
    df['dataset'] = df['Context'] + ' ' + df['Response']
    return df


def main():
    df = load_data()
    doc2vec = Doc2vec(df, 'dataset')
    doc2vec.save('.cache/model_doc2vec')

    word2vec = Word2vec(df, 'dataset')
    word2vec.save('.cache/model_word2vec')

    tfidf = Tfidf(df, 'dataset')
    tfidf.save('.cache/model_tfidf')


if __name__ == '__main__':
    main()
