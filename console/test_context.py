import logging

import pandas as pd

from rc_modules import Proofing

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
logging.getLogger().setLevel(logging.CRITICAL)


def load_data():
    url = 'https://raw.githubusercontent.com/rochimfn/tanyahukum-bot/main/dataset.csv'
    df = pd.read_csv(url, sep='\t', quotechar='\'')
    df['dataset'] = df['Context'] + ' ' + df['Keywords'] + ' ' + df['Response']
    return df


def main():
    proofing = Proofing()
    df = load_data()
    for index, question in df['Context'].iteritems():
        try:
            proofing.check_words(question.lower())
        except ValueError as e:
            unpassed = df.at[index,'Context']
            problem = e.args[0]
            print(f'index: {index} \nquestion: {unpassed} \nproblem: {problem}')


if __name__ == '__main__':
    main()
