import logging
import pickle
from pathlib import Path
from typing import Optional, Any, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Tfidf:
    def __init__(self, cache: Optional[str] = None):
        self.model: Optional[TfidfVectorizer] = None
        self.dataframe: Optional[pd.DataFrame] = None
        self.dataset_key: Optional[str] = None
        self.matrix: Optional[Any] = None
        if cache is not None:
            self.__load_cache(cache)

    def __load_cache(self, path: str):
        dataframe_path = Path(path, 'tfidf_dataframe')
        dataframe_exist = dataframe_path.is_file()

        dataset_key_path = Path(path, 'tfidf_dataset_key')
        dataset_key_exist = dataset_key_path.is_file()

        model_path = Path(path, 'tfidf_model')
        model_exist = model_path.is_file()

        matrix_path = Path(path, 'tfidf_matrix')
        matrix_exist = matrix_path.is_file()
        if not (dataframe_exist and dataset_key_exist and model_exist and matrix_exist):
            raise Exception('Cache not found')

        self.dataframe: pd.DataFrame = pd.read_pickle(dataframe_path)
        with open(dataset_key_path, 'rb') as fd:
            self.dataset_key: str = pickle.load(fd)
        with open(model_path, 'rb') as fd:
            self.model: TfidfVectorizer = pickle.load(fd)
        with open(matrix_path, 'rb') as fd:
            self.matrix: Any = pickle.load(fd)

    def set_dataframe(self, dataframe: pd.DataFrame, dataset_key: str = 'dataset'):
        if self.model is None:
            self.dataframe = dataframe.copy()
            self.dataset_key = dataset_key
        else:
            raise Exception(
                'Cannot set dataframe when model is already fitted')

    def __train(self, retrain=False):
        if self.dataframe is None or self.dataset_key is None:
            raise Exception('Dataframe and dataset key is None')

        if retrain or self.model is None:
            self.model = TfidfVectorizer()
            self.matrix = self.model.fit_transform(
                self.dataframe[self.dataset_key])

    def infer_vector(self, sentence: List[str]):
        self.__train()
        if not isinstance(sentence, list):
            raise Exception("Wrong sentence data type! Use <class 'list'>")
        return self.model.transform(sentence)

    def ask(self, query: str, num_rank=10):
        query_vec = self.infer_vector([query])
        results = cosine_similarity(self.matrix, query_vec).reshape((-1,))
        self.dataframe['Similarity'] = results.tolist()
        top_idx = results.argsort()[-1 * num_rank:][::-1]

        return self.dataframe.loc[top_idx, :]

    def create_cache(self, path: str):
        cache_path = Path(path)
        path_not_exist = not cache_path.is_dir()
        if path_not_exist:
            cache_path.mkdir(parents=True, exist_ok=True)

        self.__train()
        with open(Path(path, 'tfidf_model'), 'wb') as fd:
            pickle.dump(self.model, fd)
        self.dataframe.to_pickle(str(Path(path, 'tfidf_dataframe')))
        with open(Path(path, 'tfidf_dataset_key'), 'wb') as fd:
            pickle.dump(self.dataset_key, fd)
        with open(Path(path, 'tfidf_matrix'), 'wb') as fd:
            pickle.dump(self.matrix, fd)
