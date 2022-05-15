import logging
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, Response, status

from rc_modules import Doc2vec, Tfidf, Word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

app = FastAPI()

tfidf = Tfidf(cache='.cache/tfidf')
word2vec = Word2vec(cache='.cache/word2vec')
doc2vec = Doc2vec(cache='.cache/doc2vec')


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
async def ask(response: Response, algorithm: str, q: Optional[str] = None):
    answer = pd.DataFrame()
    if algorithm not in ('tfidf', 'word2vec', 'doc2vec'):
        response.status_code = status.HTTP_400_BAD_REQUEST
        return fail_response({'algorithm': 'Supported algorithm: tfidf, word2vec, doc2vec'})
    if q is None:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return fail_response({'q': 'Question is required!'})
    if algorithm == 'tfidf':
        answer: pd.DataFrame = tfidf.ask(query=q, num_rank=1)
    elif algorithm == 'word2vec':
        answer: pd.DataFrame = word2vec.ask(query=q, num_rank=1)
    elif algorithm == 'doc2vec':
        answer: pd.DataFrame = doc2vec.ask(query=q, num_rank=1)
    if not answer.empty:
        answer['Rank'] = answer.reset_index().index + 1
        data = {
            'question': q,
            'answer': answer.iloc[[0]]['Response'].item()
        }
        return success_response(data)


if __name__ == "__main__":
    uvicorn.run("web:app", host="0.0.0.0", port=8000, log_config=None)
