import logging
from typing import TypedDict, Optional, List

import pandas as pd
import streamlit as st

from rc_modules import Proofing, Doc2vec, Tfidf, Word2vec, ReturnType

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


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
    doc2vec = Doc2vec(cache='.cache/doc2vec')
    word2vec = Word2vec(cache='.cache/word2vec')

    question = st.text_input(
        label='Masukkan pertanyaan tentang konsititusi',
        value='Apa tugas lembaga negara')
    show_tfidf = st.checkbox('Tfidf', value=True)
    show_doc2vec = st.checkbox('Doc2vec', value=True)
    show_word2vec = st.checkbox('Word2vec', value=True)

    if st.button('Tanyakan'):
        result = pre_process(question)
        if result['correct'] is not True:
            handle_not_proof(question, result['false_word'])
        else:
            if show_tfidf:
                answer: pd.DataFrame = tfidf.ask(question)
                answer['Rank'] = answer.reset_index().index + 1
                st.subheader('Tfidf')
                st.table(answer[['Rank', 'Response', 'Similarity']])

            if show_doc2vec:
                answer: pd.DataFrame = doc2vec.ask(question)
                answer['Rank'] = answer.reset_index().index + 1
                st.subheader('Doc2vec')
                st.table(answer[['Rank', 'Response', 'Similarity']])

            if show_word2vec:
                answer: pd.DataFrame = word2vec.ask(question)
                answer['Rank'] = answer.reset_index().index + 1
                st.subheader('Word2vec')
                st.table(answer[['Rank', 'Response', 'Similarity']])


if __name__ == '__main__':
    main()
