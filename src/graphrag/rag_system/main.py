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
