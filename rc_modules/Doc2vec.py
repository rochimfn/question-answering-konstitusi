import logging
import pickle
from pathlib import Path
from typing import Iterable, TypedDict, Optional, List

import pandas as pd
from gensim.models import doc2vec
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class Doc2vecOptions(TypedDict):
    vector_size: int
    min_count: int
    epochs: int


class Doc2vec:
    def __init__(self, cache: Optional[str] = None):
        self.model: Optional[doc2vec.Doc2Vec] = None
        self.dataframe: Optional[pd.DataFrame] = None
        self.dataset_key: Optional[str] = None
        if cache is not None:
            self.__load_cache(cache)
        self.options: Doc2vecOptions = {
            'vector_size': 50,
            'min_count': 2,
            'epochs': 40
        }

    def __load_cache(self, path: str):
        dataframe_path = Path(path, 'doc2vec_dataframe')
        dataframe_exist = dataframe_path.is_file()

        dataset_key_path = Path(path, 'doc2vec_dataset_key')
        dataset_key_exist = dataset_key_path.is_file()

        model_path = Path(path, 'doc2vec_model')
        model_exist = model_path.is_file()
        if not (dataframe_exist and dataset_key_exist and model_exist):
            raise Exception('Cache not found')

        self.dataframe: pd.DataFrame = pd.read_pickle(dataframe_path)
        with open(dataset_key_path, 'rb') as fd:
            self.dataset_key: str = pickle.load(fd)
        self.model: doc2vec.Doc2Vec = doc2vec.Doc2Vec.load(str(model_path))

    def set_dataframe(self, dataframe: pd.DataFrame, dataset_key: str = 'dataset'):
        if self.model is None:
            self.dataframe = dataframe.copy()
            self.dataset_key = dataset_key
        else:
            raise Exception(
                'Cannot set dataframe when model is already fitted')

    def set_options(self, options: Doc2vecOptions):
        self.options = options

    def __read_corpus(self, corpus: Iterable):
        for i, line in enumerate(corpus):
            tokens = simple_preprocess(line)
            yield doc2vec.TaggedDocument(tokens, [i])

    def __train(self, retrain=False):
        if self.dataframe is None or self.dataset_key is None:
            raise Exception('Dataframe and dataset key is None')

        if retrain or self.model is None:
            train_corpus = list(self.__read_corpus(
                self.dataframe[self.dataset_key]))
            model = doc2vec.Doc2Vec(
                vector_size=self.options['vector_size'],
                min_count=self.options['min_count'],
                epochs=self.options['epochs'])
            model.build_vocab(train_corpus)
            model.train(train_corpus, total_examples=model.corpus_count,
                        epochs=model.epochs)
            self.model = model

    def infer_vector(self, sentence: List[str]):
        self.__train()
        if not isinstance(sentence, list):
            raise Exception("Wrong sentence data type! Use <class 'list'>")
        return self.model.infer_vector(sentence)

    def __gen_vector(self):
        self.__train()
        self.dataframe['vector'] = self.dataframe[self.dataset_key].apply(
            lambda x: self.infer_vector(x.split(' ')))

    def ask(self, query: str, num_rank=10):
        if not isinstance(query, str):
            raise Exception("Wrong query data type! Use 'str'")

        self.__gen_vector()
        self.dataframe['Similarity'] = self.dataframe['vector'].apply(lambda x: cosine_similarity(
            x.reshape(1, -1), self.infer_vector(query.split(' ')).reshape(1, -1)).item())

        return self.dataframe.sort_values(
            by='Similarity', ascending=False).head(num_rank).copy()

    def create_cache(self, path: str):
        cache_path = Path(path)
        path_not_exist = not cache_path.is_dir()
        if path_not_exist:
            cache_path.mkdir(parents=True, exist_ok=True)

        self.__train()
        with open(Path(path, 'doc2vec_model'), 'wb') as fd:
            self.model.save(fd)
        self.dataframe.to_pickle(str(Path(path, 'doc2vec_dataframe')))
        with open(Path(path, 'doc2vec_dataset_key'), 'wb') as fd:
            pickle.dump(self.dataset_key, fd)
