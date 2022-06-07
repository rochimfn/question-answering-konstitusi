import logging
import os
from typing import List, Optional, TypedDict

import pandas as pd
import streamlit as st

from rc_modules import Doc2vec, Proofing, ReturnType, Tfidf, Word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


ENABLE_PROOFING = os.getenv('ENABLE_PROOFING', False)


class PreProcessResult(TypedDict):
    correct: bool
    false_word: List[Optional[ReturnType]]


def pre_process(question: str) -> PreProcessResult:
    p = Proofing()
    result:  List[Optional[ReturnType]] = p.check_words(question)
    false_word = list(filter(lambda x: x['exists'] is False, result))
    return {'correct': len(false_word) == 0, 'false_word': false_word}


def handle_not_proof(question: str, false_word: List[Optional[ReturnType]]):
    st.write(f'Pertanyaan anda terdeteksi tidak valid: {question}')
    for word in false_word:
        suggestion = ', '.join(word['suggestion'])
        st.write('Kata "{}" tidak ditemukan, saran: {}'
                 .format(word['word'], suggestion))


def main():
    st.set_page_config(layout="wide")
    st.title('Development tugas akhir')

    tfidf = Tfidf(cache='.cache/tfidf')
    word2vec = Word2vec(cache='.cache/word2vec')
    doc2vec = Doc2vec(cache='.cache/doc2vec')

    question = st.text_input(
        label='Masukkan pertanyaan tentang konsititusi',
        value='Apa tugas lembaga negara')
    show_tfidf = st.checkbox('Tfidf', value=True)
    show_word2vec = st.checkbox('Word2vec', value=True)
    show_doc2vec = st.checkbox('Doc2vec', value=True)

    if st.button('Tanyakan'):
        result = pre_process(question) if ENABLE_PROOFING else False
        if ENABLE_PROOFING and result['correct'] is not True:
            handle_not_proof(question, result['false_word'])
            show_tfidf = False
            show_doc2vec = False
            show_word2vec = False

        if show_tfidf:
            answer: pd.DataFrame = tfidf.ask(question)
            answer['Rank'] = answer.reset_index().index + 1
            st.subheader('Tfidf')
            st.table(answer[['Rank', 'Response', 'Similarity']])

        if show_word2vec:
            answer: pd.DataFrame = word2vec.ask(question)
            answer['Rank'] = answer.reset_index().index + 1
            st.subheader('Word2vec')
            st.table(answer[['Rank', 'Response', 'Similarity']])

        if show_doc2vec:
            answer: pd.DataFrame = doc2vec.ask(question)
            answer['Rank'] = answer.reset_index().index + 1
            st.subheader('Doc2vec')
            st.table(answer[['Rank', 'Response', 'Similarity']])


if __name__ == '__main__':
    main()
