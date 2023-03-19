import nltk
import re

nltk.download('wordnet')


stpowords = nltk.corpus.stopwords.words('russian') + nltk.corpus.stopwords.words('english')
stemmer_ru = nltk.stem.SnowballStemmer("russian")
stemmer_en = nltk.stem.SnowballStemmer("english")
stemmer = lambda x: stemmer_en.stem(x) if re.match(r"[a-zA-Z]", x, flags=0) else stemmer_ru.stem(x)


def fix_string(string):
    string = re.sub('[^a-zA-Zа-яА-Я0-9]', ' ', string)
    string = " ".join([stemmer(word) for word in string.lower().split() if word not in stpowords and len(word) > 2])

    return string

