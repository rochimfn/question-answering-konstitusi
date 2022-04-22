from difflib import get_close_matches
from pathlib import Path
import requests
import json


class Proofing:
    dict_file = Path('.cache/dictionary.txt')
    dict_url = 'https://raw.githubusercontent.com/rochimfn/' + \
        'kamus-kata-indonesia/main/manual-kamus-indonesia.txt'
    dict = None

    custom_dict_file = Path('.cache/custom_dict.json')
    custom_dict_url = 'https://gist.githubusercontent.com/rochimfn/' + \
        '07e9c789ab1effb1de262e2d065ab400/' + \
        'raw/91367da962eba01a5176c9cf2b425f876e77086f/custom_dict.json'
    custom_dict = None

    def __init__(self):
        if not (self.dict_file.is_file() and self.custom_dict_file.is_file()):
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

    def __load_dict(self):
        with open(self.dict_file, 'r') as f:
            self.dict = f.read().split('\n')
        with open(self.custom_dict_file, 'r') as f:
            self.custom_dict = json.loads(f.read())

    def suggest(self, word):
        dict_subset = [w for w in self.dict if w.startswith(word[0])]
        return get_close_matches(word, dict_subset, cutoff=0.8)

    def check_word(self, word):
        if len(word) == 0:
            return {'word': word, 'exists': False, 'suggestion': []}

        if word in self.custom_dict:
            word = self.custom_dict[word]

        if word.lower() in self.dict:
            return {'word': word, 'exists': True}
        else:
            suggestion = self.suggest(word.lower())
            return {'word': word, 'exists': False, 'suggestion': suggestion}

    def check_words(self, words):
        if len(words) == 0:
            return {'exists': False, 'suggestion': []}
        return [self.check_word(word) for word in words.split(' ')]


if __name__ == '__main__':
    p = Proofing()
    print(f'Length of dictionary {len(p.dict)}')

    word = 'Pulang'
    print(f'Word {word} in dictionary: {p.check_word(word)}')

    false_word = 'Pulagn'
    print(f'Word {false_word} in dictionary: {p.check_word(false_word)}')

    sentence = 'Bapak pulang kampung'
    print(f'Kalimat : {sentence}')
    print(f'Hasil: {p.check_words(sentence)}')

    false_sentence = 'Bapak pulagn kampung'
    print(f'Kalimat : {false_sentence}')
    print(f'Hasil: {p.check_words(false_sentence)}')

    another_sentence = 'bgmn bisa'
    print(f'Kalimat : {another_sentence}')
    print(f'Hasil: {p.check_words(another_sentence)}')

    another_false_sentence = 'bg bisa'
    print(f'Kalimat : {another_false_sentence}')
    print(f'Hasil: {p.check_words(another_false_sentence)}')
