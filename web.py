import logging
from os import environ
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, Response, status

from rc_modules import Doc2vec, Tfidf, Word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

QA_HOST = environ['QA_HOST']
QA_PORT = int(environ['QA_PORT'])

app = FastAPI()

models = {
    'tfidf': Tfidf(cache='.cache/tfidf'),
    'doc2vec': Doc2vec(cache='.cache/doc2vec'),
    'word2vec': Word2vec(cache='.cache/word2vec')
}


@app.get('/')
async def root():
    return {'status': 'success',
            'data': []}


def success_response(data: dict) -> dict:
    return {'status': 'success',
            'data': data}


def fail_response(data: dict) -> dict:
    return {'status': 'fail',
            'data': data}


@app.get('/{algorithm}/', status_code=200)
async def ask(response: Response, algorithm: str, q: Optional[str] = None, num_rank: Optional[int] = 10):
    if algorithm not in ('tfidf', 'word2vec', 'doc2vec'):
        response.status_code = status.HTTP_400_BAD_REQUEST
        return fail_response({'algorithm': 'Supported algorithm: tfidf, word2vec, doc2vec'})

    if q is None:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return fail_response({'q': 'Question is required!'})

    answer: pd.DataFrame = models[algorithm].ask(query=q, num_rank=num_rank)
    if not answer.empty:
        data = {
            'question': q,
            'answer': answer['Response'].to_list()
        }
        return success_response(data)


if __name__ == "__main__":
    uvicorn.run("web:app", host=QA_HOST, port=QA_PORT, log_config=None)
