import json
from difflib import get_close_matches
from pathlib import Path
from typing import List, Optional

import requests
from gensim.utils import simple_preprocess


class Proofing:
    dict_file = Path('.cache/dictionary.txt')
    dict_url = 'https://raw.githubusercontent.com/rochimfn/' + \
               'kamus-kata-indonesia/main/manual-kamus-indonesia.txt'
    dict: Optional[List[str]] = None

    custom_dict_file = Path('.cache/custom_dict.json')
    custom_dict_url = 'https://gist.githubusercontent.com/rochimfn/' + \
                      '07e9c789ab1effb1de262e2d065ab400/' + \
                      'raw/9e4c557d133cd3fad1c546dca1424ef56098da1c/custom_dict.json'
    custom_dict: Optional[List[str]] = None

    context_dict_file = Path('.cache/konstitusi.json')
    context_dict_url = 'https://gist.githubusercontent.com/rochimfn/' + \
        '07e9c789ab1effb1de262e2d065ab400/' + \
        'raw/9e4c557d133cd3fad1c546dca1424ef56098da1c/konstitusi.json'
    context_dict: Optional[List[str]] = None

    def __init__(self):
        if not (self.dict_file.is_file() and self.custom_dict_file.is_file()
                and self.context_dict_file.is_file()):
            self.__download_dict()
        self.__load_dict()

    def __download_dict(self):
        r = requests.get(self.dict_url)
        with open(self.dict_file, 'wb') as f:
            if r.status_code == 200:
                f.write(r.content)

        r = requests.get(self.custom_dict_url)
        with open(self.custom_dict_file, 'wb') as f:
            if r.status_code == 200:
                f.write(r.content)

        r = requests.get(self.context_dict_url)
        with open(self.context_dict_file, 'wb') as f:
            if r.status_code == 200:
                f.write(r.content)

    def __load_dict(self):
        with open(self.dict_file, 'r') as f:
            self.dict = f.read().split('\n')
        with open(self.custom_dict_file, 'r') as f:
            self.custom_dict = json.loads(f.read())
        with open(self.context_dict_file, 'r') as f:
            self.context_dict = json.loads(f.read())

    def suggest(self, word: str) -> List[Optional[str]]:
        dict_subset = [w for w in self.dict if w.startswith(word[0])]
        return get_close_matches(word, dict_subset, cutoff=0.8)

    def check_word(self, word: str) -> str:
        if len(word) == 0:
            raise ValueError('Kata tidak boleh kosong!')

        if word.lower() in self.dict:
            return word.lower()
        elif word.lower() in self.custom_dict:
            return self.custom_dict[word]
        else:
            suggestion = ', '.join(self.suggest(word.lower()))
            raise ValueError(
                f'Kata "{word}" tidak ditemukan! \nMungkin maksud anda: {suggestion}')

    def check_words(self, words: str) -> str:
        if len(words) == 0:
            return []
        if self.__fit_context(words):
            return ' '.join([self.check_word(word) for word in simple_preprocess(words)])
        else:
            raise ValueError(f'"{words}" terdeteksi bukan pertanyaan tentang konstitusi Indonesia.')

    def __fit_context(self, question: str) -> bool:
        fit = False
        for word in simple_preprocess(question):
            if word in self.context_dict:
                fit = True
                break
        
        return fit
        


if __name__ == '__main__':
    p = Proofing()
    print(f'Length of dictionary {len(p.dict)}')

    word = 'Pulang'
    print(f'Word {word} in dictionary: {p.check_word(word)}')

    false_word = 'Pulagn'
    try:
        print(f'Word {false_word} in dictionary: {p.check_word(false_word)}')
    except ValueError as e:
        print(e.args[0])

    sentence = 'Bapak pulang kampung'
    print(f'Kalimat : {sentence}')
    print(f'Hasil: {p.check_words(sentence)}')

    try:
        false_sentence = 'Bapak pulagn kampung'
        print(f'Kalimat : {false_sentence}')
        print(f'Hasil: {p.check_words(false_sentence)}')
    except ValueError as e:
        print(e.args[0])

    another_sentence = 'bgmn bisa'
    print(f'Kalimat : {another_sentence}')
    print(f'Hasil: {p.check_words(another_sentence)}')

    try:
        another_false_sentence = 'bg bisa'
        print(f'Kalimat : {another_false_sentence}')
        print(f'Hasil: {p.check_words(another_false_sentence)}')
    except ValueError as e:
        print(e.args[0])
