import nltk
from rake_nltk import Rake
from keybert import KeyBERT
import time

text = 'Natural language processing (NLP) is a subfield of artificial intelligence (AI) focused on the interaction between computers and humans through natural language.' \
' The ultimate objective of NLP is to enable computers to understand, interpret, and generate human language in a way that is valuable.' \
' Applications of NLP include sentiment analysis, machine translation, chatbots, and information retrieval.'

def keywords_extractor(text: str, n=3, method='rake') -> list[str]:
    if method not in ['rake', 'bert']:
        raise ValueError("Method must be either 'rake' or 'bert'")
    
    if method == 'rake':
        r = Rake()
        r.extract_keywords_from_text(text)
        phrases = r.get_ranked_phrases()[:n]
        words = set()
        for phrase in phrases:
            for word in phrase.split():
                words.add(word.lower())
        return list(words)
    
    if method == 'bert':
        kwBert = KeyBERT()
        keywords = kwBert.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=3)
        kw = []
        for k in keywords:
            kw.append(k[0])
        return kw    

if __name__ == '__main__':
    start = time.perf_counter()
    print('RAKE:', keywords_extractor(text))
    end = time.perf_counter()
    print(f'RAKE Time: {end - start:.4f} seconds')
    