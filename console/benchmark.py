import logging
import pandas as pd

from pathlib import Path
from rc_modules import Doc2vec, Tfidf, Word2vec
from sys import exit

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

def load_query_set():
    url = 'https://raw.githubusercontent.com/rochimfn/tanyahukum-bot/main/dataset.csv'
    df = pd.read_csv(url, sep='\t', quotechar='\'')
    df['dataset'] = df['Context'] + ' ' + df['Response']
    return list(df['Context'])

def main():
    save_dir = input('Simpan hasil ke direktori: ')
    save_path = Path(save_dir)
    if save_path.is_dir:
        logging.info('Direktori valid, tunggu sebentar :)')
    else:
        logging.error('Direktori tidak valid')
        exit(1)

    doc2vec = Doc2vec()
    doc2vec.load('.cache/model_doc2vec')

    word2vec = Word2vec()
    word2vec.load('.cache/model_word2vec')

    tfidf = Tfidf()
    tfidf.load('.cache/model_tfidf')

    query_set = load_query_set()
    writer_doc2vec = pd.ExcelWriter(save_path / 'doc2vec.xlsx', engine='xlsxwriter')
    writer_word2vec = pd.ExcelWriter(save_path / 'word2vec.xlsx', engine='xlsxwriter')
    writer_tfidf = pd.ExcelWriter(save_path / 'tfidf.xlsx', engine='xlsxwriter')

    for index, question in enumerate(query_set):
        answer = doc2vec.ask(question)
        answer['Rank'] = answer.reset_index().index + 1
        answer[['Rank','Response']].to_excel(writer_doc2vec, sheet_name=f'{index}. {question}'[:31])

        answer = word2vec.ask(question)
        answer['Rank'] = answer.reset_index().index + 1
        answer[['Rank', 'Response']].to_excel(writer_word2vec, sheet_name=f'{index}. {question}'[:31])

        answer = tfidf.ask(question)
        answer['Rank'] = answer.reset_index().index + 1
        answer[['Rank', 'Response']].to_excel(writer_tfidf, sheet_name=f'{index}. {question}'[:31])
    
    writer_doc2vec.save()
    writer_word2vec.save()
    writer_tfidf.save()

if __name__ == '__main__':
    main()
