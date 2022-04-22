from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import logging
import pandas as pd
import pickle

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Tfidf:
    def __init__(self, dataframe: pd.DataFrame = None, dataset_key: str = 'dataset'):
        self.dataframe = dataframe.copy() if dataframe is not None else None
        self.dataset_key = dataset_key
        self.model = None
        self.matrix = None

    def __train(self, retrain=False):
        if retrain or self.model is None:
            self.model = TfidfVectorizer()
            self.matrix = self.model.fit_transform(
                self.dataframe[self.dataset_key])

    def infer_vector(self, sentence):
        self.__train()
        words = []
        if isinstance(sentence, str):
            words = [sentence]
        elif isinstance(sentence, list):
            words = sentence
        return self.model.transform(words)

    def ask(self, query, num_rank=10):
        query_vec = self.infer_vector(query)
        results = cosine_similarity(self.matrix, query_vec).reshape((-1,))
        self.dataframe['Similarity'] = results.tolist()
        top_idx = results.argsort()[-1*num_rank:][::-1]

        return self.dataframe.loc[top_idx, :]

    def __save_dataframe(self, path):
        self.dataframe.to_pickle(path + '_dataframe')
        with open(Path(path + '_dataset_key'), 'wb') as fd:
            pickle.dump(self.dataset_key, fd)

    def __save_model(self, path):
        with open(Path(path), 'wb') as fd:
            pickle.dump(self.model, fd)
        with open(Path(path+'_matrix'), 'wb') as fd:
            pickle.dump(self.matrix, fd)

    def save(self, path):
        self.__train()
        self.__save_model(path)
        self.__save_dataframe(path)

    def __load_dataframe(self, path):
        self.dataframe = pd.read_pickle(path + '_dataframe')
        with open(Path(path + '_dataset_key'), 'rb') as fd:
            self.dataset_key = pickle.load(fd)

    def __load_model(self, path):
        with open(Path(path), 'rb') as fd:
            self.model = pickle.load(fd)
        with open(Path(path+'_matrix'), 'rb') as fd:
            self.matrix = pickle.load(fd)

    def load(self, path):
        self.__load_dataframe(path)
        self.__load_model(path)
