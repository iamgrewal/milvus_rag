import spacy
import re
import nltk
from nltk.corpus import stopwords
from graphrag.logger import logger
from collections import defaultdict

nltk.download('stopwords', quiet=True)

class NLPProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return ' '.join([w for w in text.split() if w.lower() not in self.stop_words])

    def extract_entities_and_relations(self, text):
        doc = self.nlp(text)
        entities = [(ent.text.strip(), ent.label_) for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
        relations = []
        for sent in doc.sents:
            sentence_ents = [ent for ent in entities if ent[0] in sent.text]
            for i in range(len(sentence_ents)):
                for j in range(i+1, len(sentence_ents)):
                    relations.append((sentence_ents[i], sentence_ents[j], "CO_OCCUR"))
        return entities, relations
