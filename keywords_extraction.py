import nltk
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('wordnet')
from rake_nltk import Rake
from nltk.corpus import wordnet as wn
from keybert import KeyBERT

def keywords_extractor_RAKE(text: str, n=3) -> list[str]:
    r = Rake()
    r.extract_keywords_from_text(text)
    phrases = r.get_ranked_phrases()[:n]

    words = set()
    for phrase in phrases:
        for word in phrase.split():
            words.add(word.lower())
    return list(words)


def keywords_extractor_BERT(text: str) -> list[str]:
    kwBert = KeyBERT()
    keywords = kwBert.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=3)
    return keywords