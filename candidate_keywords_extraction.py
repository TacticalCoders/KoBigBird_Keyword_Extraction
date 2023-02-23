from konlpy.tag import Komoran
from konlpy.tag import Kkma
from konlpy.tag import Okt
from collections import Counter
import nltk
from string import punctuation
import re
import stopword
import konlpy


class CandidatesExtractor:
    def __init__(self):
        self.ko_stopwords = stopword.get_ko_stopwords()
        self.eng_stopwords = stopword.get_eng_stopwords()

    def remove_stopwords(self, nouns_list, stopword_set):
        #     for noun in nouns_list[:]: # 원소를 삭제하는 과정에서 누락이 발생하기 때문에 [:]를 통해 리스트의 복사본을 주어야 함.
        #         if noun in stopword_set:
        #             nouns_list.remove(noun)
        arr_removed = [n for n in nouns_list if n not in stopword_set]
        return arr_removed

    def get_nouns_komoran(self, text, use_phrase=False):
        komoran = Komoran()
        nouns = komoran.nouns(text)
        if use_phrase:
            phrases = self.get_nouns_phrase(text)
            nouns.extend(phrases)
        eng_nouns = self.get_nouns_eng(text)
        eng_nouns = [n for n in eng_nouns if len(n) != 1]
        eng_nouns = self.remove_stopwords(eng_nouns, self.eng_stopwords)
        nouns.extend(eng_nouns)
        nouns = self.remove_stopwords(nouns, self.ko_stopwords)
        counts = Counter(nouns)
        nouns = set(nouns)
        nouns = list(nouns)

        return nouns, counts

    def get_nouns_kkma(self, text, use_phrase=False):
        kkma = Kkma()
        nouns = kkma.nouns(text)
        if use_phrase:
            phrases = self.get_nouns_phrase(text)
            nouns.extend(phrases)
        eng_nouns = self.get_nouns_eng(text)
        eng_nouns = [n for n in eng_nouns if len(n) != 1]
        eng_nouns = self.remove_stopwords(eng_nouns, self.eng_stopwords)
        nouns.extend(eng_nouns)
        nouns = self.remove_stopwords(nouns, self.ko_stopwords)
        counts = Counter(nouns)
        nouns = set(nouns)
        nouns = list(nouns)

        return nouns, counts

    def get_nouns_okt(self, text, use_phrase=False):
        okt = Okt()
        nouns = okt.nouns(text)
        if use_phrase:
            phrases = self.get_nouns_phrase(text)
            nouns.extend(phrases)
        eng_nouns = self.get_nouns_eng(text)
        eng_nouns = [n for n in eng_nouns if len(n) != 1]
        eng_nouns = self.remove_stopwords(eng_nouns, self.eng_stopwords)
        nouns.extend(eng_nouns)
        nouns = self.remove_stopwords(nouns, self.ko_stopwords)
        counts = Counter(nouns)
        nouns = set(nouns)
        nouns = list(nouns)

        return nouns, counts

    def get_nouns_phrase(self, text):

        phrases = []
        words = konlpy.tag.Komoran().pos(text)

        # Define a chunk grammar, or chunking rules, then chunk
        grammar = """
        NP: {<N.*>*<Suffix>?}   # Noun phrase
        """
        parser = nltk.RegexpParser(grammar)
        chunks = parser.parse(words)

        print("\n# Print noun phrases only")
        for subtree in chunks.subtrees():
            if subtree.label() == 'NP':
                phrases.append(' '.join((e[0] for e in list(subtree))))
        return phrases


    def get_nouns_eng(self, text):
        # 명사 추출 시 url 제거
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = re.sub(url_pattern, '', text)
        url_pattern = re.compile(r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{2,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)")
        text = re.sub(url_pattern, '', text)

        text = text.encode('utf-8').decode('ascii', 'ignore')

        result = ""
        for t in text:
            if t not in punctuation:
                result += t

        text = result.strip()

        word_tokens = nltk.word_tokenize(text)
        tokens_pos = nltk.pos_tag(word_tokens)
        NN_words = []
        for word, pos in tokens_pos:
            if 'NN' in pos or 'NNP' in pos:
                NN_words.append(word)

        return NN_words
