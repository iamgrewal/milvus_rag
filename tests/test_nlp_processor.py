import unittest
from graphrag.nlp.processor import NLPProcessor
class TestNLPProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = NLPProcessor()

    def test_preprocess(self):
        raw_text = "Apple was founded in 1976."
        processed = self.processor.preprocess(raw_text)
        self.assertNotIn("was", processed)

    def test_entity_extraction(self):
      text = "Steve Jobs founded Apple."
      entities, relations = self.processor.extract_entities_and_relations(text)
      entity_names = [e for e, _ in entities]
      self.assertIn("Steve Jobs", entity_names)
      self.assertIn("Apple", entity_names)

