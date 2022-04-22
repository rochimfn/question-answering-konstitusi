from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import logging
import numpy as np
import pandas as pd
import pickle


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class Word2vec:
    default_options = {
        'size': 50,
        'min_count': 2,
        'iter': 40
    }

    def __init__(self, dataframe: pd.DataFrame = None,
                 dataset_key: str = 'dataset',
                 option=default_options):
        self.dataframe = dataframe.copy() if dataframe is not None else None
        self.dataset_key = dataset_key
        self.options = option
        self.model = None

    def __read_corpus(self, corpus):
        for line in corpus:
            yield gensim.utils.simple_preprocess(line)

    def __train(self, retrain=False):
        if retrain or self.model is None:
            train_corpus = list(self.__read_corpus(
                self.dataframe[self.dataset_key]))
            model = gensim.models.Word2Vec(
                train_corpus,
                size=self.options['size'],
                min_count=self.options['min_count'],
                iter=self.options['iter'])
            self.model = model

    def __infer_vector(self, sentences):
        if len(sentences) < 1:
            return np.zeros(50)
        else:
            vector = []
            for word in sentences:
                if word in self.model.wv:
                    vector.append(self.model.wv[word])
                else:
                    vector.append(np.zeros(50))

            return np.mean(vector, axis=0)

    def __gen_vector(self):
        self.__train()
        self.dataframe['vector'] = self.dataframe[self.dataset_key].apply(
            lambda x: self.__infer_vector(x.split(' ')))

    def infer_vector(self, sentence):
        self.__train()
        words = []
        if isinstance(sentence, str):
            words = sentence.split()
        elif isinstance(sentence, list):
            words = sentence
        return self.__infer_vector(words)

    # def ask(self, query, num_rank=10):
    #     words = []
    #     if isinstance(query, str):
    #         words = query.split()
    #     elif isinstance(query, list):
    #         words = query
    #     else:
    #         return None

    #     self.__gen_vector()
    #     index = gensim.similarities.SparseMatrixSimilarity(
    #         self.dataframe['vector'],
    #         num_features=self.options['size'])

    #     sims = index[self.__infer_vector(words)]
    #     top_idx = sims.argsort()[-1*num_rank:][::-1]

    #     return self.dataframe.loc[top_idx, :]

    def ask(self, query, num_rank=10):
        words = []
        if isinstance(query, str):
            words = query.split()
        elif isinstance(query, list):
            words = query
        else:
            return None

        self.__gen_vector()

        self.dataframe['Similarity'] = self.dataframe['vector'].apply(lambda x: cosine_similarity(
            x.reshape(1, -1), self.infer_vector(words).reshape(1, -1)).item())

        return self.dataframe.sort_values(
            by='Similarity', ascending=False).head(num_rank).copy()

    def __save_dataframe(self, path):
        self.dataframe.to_pickle(path + '_dataframe')
        with open(Path(path + '_dataset_key'), 'wb') as fd:
            pickle.dump(self.dataset_key, fd)

    def __save_model(self, path):
        with open(Path(path), 'wb') as fd:
            self.model.save(fd)

    def save(self, path):
        self.__train()
        self.__save_model(path)
        self.__save_dataframe(path)

    def __load_dataframe(self, path):
        self.dataframe = pd.read_pickle(path + '_dataframe')
        with open(Path(path + '_dataset_key'), 'rb') as fd:
            self.dataset_key = pickle.load(fd)

    def __load_model(self, path):
        self.model = gensim.models.Word2Vec.load(path)

    def load(self, path):
        self.__load_model(path)
        self.__load_dataframe(path)
