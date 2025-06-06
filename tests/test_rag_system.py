import unittest
from graphrag.rag_system.main import RAGSystem

class TestRAGSystem(unittest.TestCase):
    def setUp(self):
        self.rag = RAGSystem()

    def test_ingest_and_answer(self):
        doc = "Apple was founded by Steve Jobs."
        self.rag.ingest(doc)

        related = self.rag.neo4j.get_related("Apple")
        self.assertIn("Steve Jobs", related)

        answer = self.rag.answer("Who founded Apple?")
        self.assertIn("Steve Jobs", answer)
