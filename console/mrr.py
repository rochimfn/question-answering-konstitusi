import logging
import json
import os
from typing import List

import pandas as pd
import requests
from rc_modules import Doc2vec, Tfidf, Word2vec

NUM_RANK = int(os.getenv('NUM_RANK', 5))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.ERROR)


def load_questions() -> List[str]:
    url = 'https://raw.githubusercontent.com/rochimfn/tanyahukum-bot/main/dataset.csv'
    df = pd.read_csv(url, sep='\t', quotechar='\'')
    return df['Context'].tolist()


def load_test_questions() -> List[str]:
    url = 'https://gist.githubusercontent.com/rochimfn/' + \
        '07e9c789ab1effb1de262e2d065ab400/raw/' + \
        'ad005f7fb332b1d7e736bdbce61b1869d69d6ab7/question.json'
    r = requests.get(url)
    if r.status_code == 200:
        return json.loads(r.content)
    else:
        raise Exception('Fail to load test questions')


def get_mrr(questions: List[str]) -> dict:
    models = {
        'tfidf': Tfidf(cache='.cache/tfidf'),
        'word2vec': Word2vec(cache='.cache/word2vec'),
        'doc2vec': Doc2vec(cache='.cache/doc2vec')
    }

    def f_get_mrr(index, question, model):
        answer: pd.DataFrame = model.ask(question, num_rank=NUM_RANK)
        if index in answer.index.to_list():
            return 1/(answer.index.to_list().index(index)+1)

        return 0

    results = {
        'tfidf': [],
        'word2vec': [],
        'doc2vec': []
    }
    for index, question in enumerate(questions):
        for key, model in models.items():
            results[key].append(f_get_mrr(index, question, model))

    return results


def calculate_mrr(list_of_value: List[int]) -> int:
    return sum(list_of_value) / len(list_of_value)


def main():
    questions = load_questions()
    test_questions = load_test_questions()

    print('Dataset:')
    for key, value in get_mrr(questions).items():
        print(f'mrr {key}: {calculate_mrr(value)}')

    print('\nTesting:')
    for key, value in get_mrr(test_questions).items():
        print(f'mrr {key}: {calculate_mrr(value)}')


if __name__ == '__main__':
    main()
