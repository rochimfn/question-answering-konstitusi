import streamlit as st
import logging

from rc_modules import Proofing, Doc2vec, Tfidf, Word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def pre_process(question):
    p = Proofing()
    result = p.check_words(question)
    false_word = list(filter(lambda x: x['exists'] is False, result))
    return {'correct': len(false_word) == 0, 'false_word': false_word}


def handle_not_proof(question, false_word):
    st.write(f'Pertanyaan anda terdeteksi tidak valid: {question}')
    for word in false_word:
        suggestion = ', '.join(word['suggestion'])
        st.write('Kata "{}" tidak ditemukan, saran: {}'
                 .format(word['word'], suggestion))


def main():
    st.set_page_config(layout="wide")
    st.title('Development tugas akhir')

    tfidf = Tfidf()
    tfidf.load('.cache/model_tfidf')
    
    doc2vec = Doc2vec()
    doc2vec.load('.cache/model_doc2vec')

    word2vec = Word2vec()
    word2vec.load('.cache/model_word2vec')


    question = st.text_input(
        label='Masukkan pertanyaan tentang konsititusi',
        value='Apa tugas lembaga negara')
    show_tfidf = st.checkbox('Tfidf')
    show_doc2vec = st.checkbox('Doc2vec')
    show_word2vec = st.checkbox('Word2vec')

    if st.button('Tanyakan'):
        result = pre_process(question)
        if result['correct'] is not True:
            handle_not_proof(question, result['false_word'])
        else:
            answer = tfidf.ask(question)
            answer['Rank'] = answer.reset_index().index + 1

            answer = doc2vec.ask(question)
            answer['Rank'] = answer.reset_index().index + 1

            answer = word2vec.ask(question)
            answer['Rank'] = answer.reset_index().index + 1

            if show_tfidf:
                st.subheader('Tfidf')
                st.table(answer[['Rank', 'Response', 'Similarity']])

            if show_doc2vec:
                st.subheader('Doc2vec')
                st.table(answer[['Rank', 'Response', 'Similarity']])

            if show_word2vec:
                st.subheader('Word2vec')
                st.table(answer[['Rank', 'Response', 'Similarity']])



if __name__ == '__main__':
    main()
