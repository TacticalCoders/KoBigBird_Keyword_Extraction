import csv
from nltk.corpus import stopwords as eng_stopwords


def get_ko_stopwords():
    stopwords = []
    with open('./data/stopwords/stopwords.txt', 'r') as f:
        for line in f:
            stopwords.append(line.rstrip('\n'))

    with open('./data/stopwords/stopwords-ko.txt', 'r') as f:
        for line in f:
            stopwords.append(line.rstrip('\n'))

    with open('./data/stopwords/stopwords-np.txt', 'r') as f:
        tsv_file = csv.reader(f, delimiter="\t")
        for line in tsv_file:
            stopwords.append(line[0])

    stopwords_set = set(stopwords)
    return stopwords_set


def get_eng_stopwords():
    return set(eng_stopwords.words('english'))
