import logging
import os

import pandas as pd
import streamlit as st

from rc_modules import Doc2vec, Proofing, Tfidf, Word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


ENABLE_PROOFING = os.getenv('ENABLE_PROOFING', 0)
NUM_RANK = int(os.getenv('NUM_RANK', 5))


def pre_process(question: str) -> str:
    p = Proofing()
    return p.check_words(question)


def main():
    st.set_page_config(layout="wide")
    st.title('Sistem Question Answering Konstitusi Indonesia')

    tfidf = Tfidf(cache='.cache/tfidf')
    word2vec = Word2vec(cache='.cache/word2vec')
    doc2vec = Doc2vec(cache='.cache/doc2vec')

    question = st.text_input(
        label='Masukkan pertanyaan tentang konsititusi',
        value='Apa tugas lembaga negara')
    show_tfidf = st.checkbox('Tfidf', value=True)
    show_word2vec = st.checkbox('Word2vec', value=True)
    show_doc2vec = st.checkbox('Doc2vec', value=True)

    if st.button('Tanyakan') or question != 'Apa tugas lembaga negara' and len(question) > 1:
        try:
            processed_question = pre_process(
                question) if ENABLE_PROOFING else question
        except ValueError as e:
            st.write(f'Pertanyaan anda terdeteksi tidak valid: {question}')
            st.write(e.args[0])
            show_tfidf = False
            show_word2vec = False
            show_doc2vec = False

        if show_tfidf:
            answer: pd.DataFrame = tfidf.ask(
                processed_question, num_rank=NUM_RANK)
            answer['Rank'] = answer.reset_index().index + 1
            st.subheader('Tfidf')
            st.table(answer[['Rank', 'Response', 'Similarity']])

        if show_word2vec:
            answer: pd.DataFrame = word2vec.ask(
                processed_question, num_rank=NUM_RANK)
            answer['Rank'] = answer.reset_index().index + 1
            st.subheader('Word2vec')
            st.table(answer[['Rank', 'Response', 'Similarity']])

        if show_doc2vec:
            answer: pd.DataFrame = doc2vec.ask(
                processed_question, num_rank=NUM_RANK)
            answer['Rank'] = answer.reset_index().index + 1
            st.subheader('Doc2vec')
            st.table(answer[['Rank', 'Response', 'Similarity']])


if __name__ == '__main__':
    main()
