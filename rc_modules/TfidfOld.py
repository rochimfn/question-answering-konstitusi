from pathlib import Path
import gensim
import logging
import pandas as pd
import pickle


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class Tfidf:
    def __init__(self, dataframe: pd.DataFrame = None, dataset_key: str='dataset'):
        self.dataframe = dataframe.copy() if dataframe is not None else None
        self.dataset_key = dataset_key
        self.corpus = None
        self.dictionary = None
        self.model = None

    def __read_dataset(self, dataset):
        return [gensim.utils.simple_preprocess(line, deacc=True) for line in dataset]

    def __train(self, retrain=False):
        if retrain or self.model is None:
            dataset = self.__read_dataset(self.dataframe[self.dataset_key])
            dictionary = gensim.corpora.Dictionary(dataset)
            corpus = [dictionary.doc2bow(line) for line in dataset]
            model = gensim.models.TfidfModel(corpus)
            self.corpus = corpus
            self.dictionary = dictionary
            self.model = model

    def ask(self, query, num_rank=10):
        words = []
        if isinstance(query, str):
            words = query.split()
        elif isinstance(query, list):
            words = query
        else:
            return None

        self.__train()
        index = gensim.similarities.SparseMatrixSimilarity(
            self.model[self.corpus], num_features=len(self.dictionary))
        query_bow = self.dictionary.doc2bow(words)

        sims = index[self.model[query_bow]]
        top_idx = sims.argsort()[-1*num_rank:][::-1]

        return self.dataframe.loc[top_idx, :]

    def __save_dataframe(self, path):
        self.dataframe.to_pickle(path + '_dataframe')
        with open(Path(path + '_dataset_key'), 'wb') as fd:
            pickle.dump(self.dataset_key, fd)

    def __save_model(self, path):
        with open(Path(path), 'wb') as fd:
            self.model.save(fd)
        with open(Path(path+'_dict'), 'wb') as fd:
            self.dictionary.save(fd)

    def save(self, path):
        self.__train()
        self.__save_model(path)
        self.__save_dataframe(path)

    def __load_dataframe(self, path):
        self.dataframe = pd.read_pickle(path + '_dataframe')
        with open(Path(path + '_dataset_key'), 'rb') as fd:
            self.dataset_key = pickle.load(fd)

    def __load_model(self, path):
        self.model = gensim.models.TfidfModel.load(path)
        self.dictionary = gensim.corpora.Dictionary.load(path+'_dict')
        dataset = self.__read_dataset(self.dataframe[self.dataset_key])
        corpus = [self.dictionary.doc2bow(line) for line in dataset]
        self.corpus = corpus

    def load(self, path):
        self.__load_dataframe(path)
        self.__load_model(path)
