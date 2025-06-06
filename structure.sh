# create_structure.sh
#!/bin/bash

mkdir -p graphrag/{config,logs,models,nlp,neo4j,milvus,rag_system,embedding_service,tests}

touch graphrag/config/config.py

cat <<EOF > graphrag/logger.py
import logging
logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GraphRAG")
EOF

touch graphrag/embedding_service/__init__.py
cat <<EOF > graphrag/embedding_service/service.py
from sentence_transformers import SentenceTransformer
from graphrag.logger import logger

class EmbeddingService:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded")

    def embed(self, text):
        return self.model.encode(text).tolist()
EOF

touch graphrag/milvus/__init__.py
cat <<EOF > graphrag/milvus/manager.py
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from graphrag.config.config import Config
from graphrag.logger import logger

class MilvusManager:
    def __init__(self):
        connections.connect(host=Config.MILVUS_HOST, port=Config.MILVUS_PORT)
        if utility.has_collection(Config.COLLECTION_NAME):
            utility.drop_collection(Config.COLLECTION_NAME)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="entity", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="entity_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=Config.EMBEDDING_DIM)
        ]
        schema = CollectionSchema(fields, description="Graph RAG Collection")
        self.collection = Collection(Config.COLLECTION_NAME, schema)
        self.collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}})
        self.collection.load()

    def insert(self, records):
        self.collection.insert(records)

    def search(self, embedding, top_k=3):
        return self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["entity"]
        )
EOF

touch graphrag/neo4j/__init__.py
cat <<EOF > graphrag/neo4j/manager.py
from neo4j import GraphDatabase
from graphrag.config.config import Config
from graphrag.logger import logger

class Neo4jManager:
    def __init__(self):
        self.driver = GraphDatabase.driver(Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD))
        logger.info("Neo4j connected")

    def create_entity_and_relation(self, entity, entity_type, related, rel_type):
        with self.driver.session() as session:
            session.write_transaction(self._create_entity_tx, entity, entity_type, related, rel_type)

    @staticmethod
    def _create_entity_tx(tx, e1, t1, e2, rel_type):
        tx.run("""
            MERGE (a:Entity {name: $e1}) SET a.type = $t1
            MERGE (b:Entity {name: $e2})
            MERGE (a)-[:RELATION {type: $rel_type}]->(b)
        """, e1=e1, t1=t1, e2=e2, rel_type=rel_type)

    def get_related(self, entity):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {name: $name})--(n) RETURN DISTINCT n.name AS name
            """, name=entity)
            return [r["name"] for r in result]
EOF

touch graphrag/nlp/__init__.py
cat <<EOF > graphrag/nlp/processor.py
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
EOF

cat <<EOF > graphrag/rag_system/main.py
import hashlib
from graphrag.logger import logger
from graphrag.milvus.manager import MilvusManager
from graphrag.neo4j.manager import Neo4jManager
from graphrag.nlp.processor import NLPProcessor
from graphrag.embedding_service.service import EmbeddingService
from transformers import pipeline

class RAGSystem:
    def __init__(self):
        self.milvus = MilvusManager()
        self.neo4j = Neo4jManager()
        self.nlp = NLPProcessor()
        self.embedder = EmbeddingService()
        self.qa_pipeline = pipeline("question-answering", model="deepset/bert-base-cased-squad2")

    def ingest(self, text):
        entities, relations = self.nlp.extract_entities_and_relations(text)
        unique = {}
        for entity, label in entities:
            if entity not in unique:
                unique[entity] = label
                vector = self.embedder.embed(entity)
                eid = int(hashlib.sha256(entity.encode()).hexdigest()[:16], 16) % (2**63)
                self.milvus.insert([{"id": eid, "entity": entity, "entity_type": label, "embedding": vector}])
        for (e1, t1), (e2, t2), rel_type in relations:
            self.neo4j.create_entity_and_relation(e1, t1, e2, rel_type)

    def answer(self, question):
        query = self.nlp.preprocess(question)
        vector = self.embedder.embed(query)
        results = self.milvus.search(vector)
        entities = [r.entity.get("entity") for r in results[0]]
        context = []
        for entity in entities:
            context.extend(self.neo4j.get_related(entity))
        context_text = ". ".join([f"{e} is related." for e in set(context)])
        return self.qa_pipeline(question=question, context=context_text)['answer']

if __name__ == "__main__":
    rag = RAGSystem()
    doc = """
    Apple Inc. was founded by Steve Jobs. Steve Jobs was succeeded by Tim Cook as CEO.
    Tim Cook lives in California. California is a US state.
    """
    rag.ingest(doc)
    q = "Who is the CEO of Apple?"
    print("Q:", q)
    print("A:", rag.answer(q))
EOF

echo "Project structure and boilerplate created in ./graphrag"
