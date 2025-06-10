"""
NLP Entity Extractor for Hybrid RAG System

This module implements entity and relationship extraction using transformer-backed NER models
with async/await patterns and metadata extraction as specified in the rhoSearcher ruleset.
"""

import asyncio
from typing import List, Dict, Any
import spacy
import structlog

logger = structlog.get_logger(__name__)


class EntityExtractor:
    """
    Production-ready entity extractor using transformer-backed NER models.
    Implements parallel entity and relation extraction with confidence scoring.
    """

    def __init__(self, model_name: str = "en_core_web_trf"):
        self.model_name = model_name
        self.nlp = None
        self.confidence_threshold = 0.7
        self.max_entity_length = 100

        # Entity type mappings for standardization
        self.entity_type_mapping = {
            "PERSON": "person",
            "ORG": "organization",
            "GPE": "location",
            "PRODUCT": "product",
            "EVENT": "event",
            "FAC": "facility",
            "LAW": "law",
            "LANGUAGE": "language",
            "WORK_OF_ART": "artwork",
            "NORP": "nationality",
            "LOC": "location",
            "MONEY": "monetary",
            "PERCENT": "percentage",
            "DATE": "temporal",
            "TIME": "temporal",
            "QUANTITY": "quantity",
            "ORDINAL": "ordinal",
            "CARDINAL": "number",
        }

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the spaCy NLP model."""
        try:
            # Load transformer-based model for better accuracy
            self.nlp = spacy.load(self.model_name)

            # Add custom pipeline components if needed
            if "entity_ruler" not in self.nlp.pipe_names:
                entity_ruler = self.nlp.add_pipe("entity_ruler", before="ner")
                # Add custom patterns for domain-specific entities
                self._add_custom_patterns(entity_ruler)

            logger.info("NLP model initialized", model=self.model_name)

        except OSError as e:
            logger.warning(
                "Transformer model not available, falling back to smaller model",
                error=str(e),
            )
            try:
                # Fallback to smaller model
                self.model_name = "en_core_web_sm"
                self.nlp = spacy.load(self.model_name)
                logger.info("Fallback NLP model loaded", model=self.model_name)
            except OSError:
                logger.error("No spaCy model available, using basic extraction")
                self.nlp = None
        except Exception as e:
            logger.error("Failed to initialize NLP model", error=str(e))
            self.nlp = None

    def _add_custom_patterns(self, entity_ruler):
        """Add custom entity patterns for domain-specific recognition."""
        try:
            custom_patterns = [
                # Technical terms
                {
                    "label": "TECHNOLOGY",
                    "pattern": [
                        {
                            "LOWER": {
                                "IN": [
                                    "ai",
                                    "ml",
                                    "nlp",
                                    "llm",
                                    "gpt",
                                    "bert",
                                    "transformer",
                                ]
                            }
                        }
                    ],
                },
                {
                    "label": "TECHNOLOGY",
                    "pattern": [{"TEXT": {"REGEX": r"^[A-Z]{2,5}$"}}],
                },  # Acronyms
                # Academic/Research terms
                {
                    "label": "RESEARCH",
                    "pattern": [
                        {"LOWER": "paper"},
                        {"LOWER": {"IN": ["on", "about"]}},
                        {"POS": "NOUN"},
                    ],
                },
                {
                    "label": "RESEARCH",
                    "pattern": [
                        {"LOWER": {"IN": ["study", "research", "analysis", "survey"]}}
                    ],
                },
                # Software/Systems
                {
                    "label": "SOFTWARE",
                    "pattern": [
                        {"TEXT": {"REGEX": r"^[a-zA-Z]+\.(py|js|cpp|java|go)$"}}
                    ],
                },  # File extensions
                {
                    "label": "SOFTWARE",
                    "pattern": [
                        {
                            "LOWER": {
                                "IN": ["framework", "library", "api", "sdk", "database"]
                            }
                        }
                    ],
                },
                # Measurements and metrics
                {
                    "label": "METRIC",
                    "pattern": [
                        {"LIKE_NUM": True},
                        {
                            "LOWER": {
                                "IN": ["gb", "mb", "kb", "ms", "sec", "min", "hour"]
                            }
                        },
                    ],
                },
                {
                    "label": "METRIC",
                    "pattern": [{"TEXT": {"REGEX": r"^\d+(\.\d+)?%$"}}],
                },  # Percentages
            ]

            entity_ruler.add_patterns(custom_patterns)
            logger.info(
                "Custom entity patterns added", pattern_count=len(custom_patterns)
            )

        except Exception as e:
            logger.error("Failed to add custom patterns", error=str(e))

    async def extract_entities_async(
        self, text: str, extract_relations: bool = True
    ) -> Dict[str, Any]:
        """
        Asynchronously extract entities and optionally relations from text.

        Args:
            text: Input text to process
            extract_relations: Whether to extract relationships between entities

        Returns:
            Dictionary containing entities, relations, and metadata
        """
        try:
            if not text or not self.nlp:
                return self._empty_extraction_result()

            # Process text in chunks if too large
            if len(text) > 10000:
                return await self._process_large_text(text, extract_relations)

            # Create processing tasks
            entity_task = asyncio.create_task(self._extract_entities(text))

            if extract_relations:
                relation_task = asyncio.create_task(self._extract_relations(text))
                entities, relations = await asyncio.gather(entity_task, relation_task)
            else:
                entities = await entity_task
                relations = []

            # Calculate extraction metadata
            metadata = self._calculate_extraction_metadata(text, entities, relations)

            result = {
                "entities": entities,
                "relations": relations,
                "metadata": metadata,
                "text_length": len(text),
                "processing_model": self.model_name,
            }

            logger.info(
                "Entity extraction completed",
                entities_count=len(entities),
                relations_count=len(relations),
                text_length=len(text),
            )

            return result

        except Exception as e:
            logger.error(
                "Entity extraction failed", error=str(e), text_length=len(text)
            )
            return self._empty_extraction_result()

    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text with confidence scoring."""
        try:
            # Process text with spaCy
            doc = self.nlp(text)

            entities = []
            seen_entities = set()  # For deduplication

            for ent in doc.ents:
                # Filter by confidence and length
                if (
                    len(ent.text) <= self.max_entity_length
                    and ent.text.strip()
                    and ent.text.lower() not in seen_entities
                ):

                    # Calculate confidence score
                    confidence = self._calculate_entity_confidence(ent, doc)

                    if confidence >= self.confidence_threshold:
                        entity_info = {
                            "text": ent.text.strip(),
                            "label": self.entity_type_mapping.get(
                                ent.label_, ent.label_.lower()
                            ),
                            "original_label": ent.label_,
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "confidence": confidence,
                            "lemma": ent.lemma_,
                            "pos_tags": [token.pos_ for token in ent],
                            "dependency": [token.dep_ for token in ent],
                            "is_alpha": ent.text.isalpha(),
                            "is_numeric": any(char.isdigit() for char in ent.text),
                            "context": self._get_entity_context(ent, doc),
                        }

                        entities.append(entity_info)
                        seen_entities.add(ent.text.lower())

            # Sort by confidence
            entities.sort(key=lambda x: x["confidence"], reverse=True)

            return entities

        except Exception as e:
            logger.error("Entity extraction failed", error=str(e))
            return []

    async def _extract_relations(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        try:
            doc = self.nlp(text)
            relations = []

            # Simple dependency-based relation extraction
            for token in doc:
                if (
                    token.dep_ in ["nsubj", "dobj", "pobj"]
                    and token.head.pos_ == "VERB"
                ):
                    # Find entities in subject-verb-object patterns
                    subject_ent = self._find_entity_for_token(token, doc.ents)
                    verb = token.head

                    # Look for object
                    for child in verb.children:
                        if child.dep_ in ["dobj", "pobj"]:
                            object_ent = self._find_entity_for_token(child, doc.ents)

                            if subject_ent and object_ent and subject_ent != object_ent:
                                relation = {
                                    "subject": subject_ent.text,
                                    "subject_type": self.entity_type_mapping.get(
                                        subject_ent.label_, subject_ent.label_.lower()
                                    ),
                                    "predicate": verb.lemma_,
                                    "object": object_ent.text,
                                    "object_type": self.entity_type_mapping.get(
                                        object_ent.label_, object_ent.label_.lower()
                                    ),
                                    "confidence": (
                                        self._calculate_entity_confidence(
                                            subject_ent, doc
                                        )
                                        + self._calculate_entity_confidence(
                                            object_ent, doc
                                        )
                                    )
                                    / 2,
                                    "sentence": token.sent.text,
                                    "relation_type": self._classify_relation_type(
                                        verb.lemma_
                                    ),
                                }
                                relations.append(relation)

            # Deduplicate relations
            unique_relations = []
            seen_relations = set()

            for rel in relations:
                rel_key = f"{rel['subject']}_{rel['predicate']}_{rel['object']}"
                if rel_key not in seen_relations:
                    seen_relations.add(rel_key)
                    unique_relations.append(rel)

            return unique_relations

        except Exception as e:
            logger.error("Relation extraction failed", error=str(e))
            return []

    def _calculate_entity_confidence(self, ent, doc) -> float:
        """Calculate confidence score for an entity."""
        try:
            base_confidence = 0.8  # Base confidence for spaCy entities

            # Adjust based on entity characteristics
            if ent.label_ in ["PERSON", "ORG", "GPE"]:
                base_confidence += 0.1  # Higher confidence for common entity types

            if ent.text.istitle():
                base_confidence += 0.05  # Title case suggests proper noun

            if len(ent.text) > 20:
                base_confidence -= 0.1  # Very long entities might be errors

            if any(char.isdigit() for char in ent.text) and ent.label_ not in [
                "DATE",
                "TIME",
                "MONEY",
                "PERCENT",
            ]:
                base_confidence -= 0.05  # Numbers in non-numeric entity types

            # Context-based adjustments
            if ent.sent:
                sentence_entities = [e for e in doc.ents if e.sent == ent.sent]
                if len(sentence_entities) > 10:
                    base_confidence -= 0.1  # Too many entities might indicate noise

            return max(0.0, min(1.0, base_confidence))

        except Exception:
            return 0.5

    def _get_entity_context(self, ent, doc, window: int = 3) -> str:
        """Get context words around an entity."""
        try:
            start_token = max(0, ent.start - window)
            end_token = min(len(doc), ent.end + window)

            context_tokens = doc[start_token:end_token]
            return " ".join([token.text for token in context_tokens])

        except Exception:
            return ""

    def _find_entity_for_token(self, token, entities):
        """Find the entity that contains a given token."""
        for ent in entities:
            if ent.start <= token.i < ent.end:
                return ent
        return None

    def _classify_relation_type(self, predicate: str) -> str:
        """Classify the type of relationship based on the predicate."""
        relation_mappings = {
            "develop": "creates",
            "create": "creates",
            "build": "creates",
            "use": "uses",
            "apply": "uses",
            "implement": "implements",
            "contain": "contains",
            "include": "contains",
            "have": "has",
            "own": "owns",
            "lead": "leads",
            "manage": "manages",
            "work": "works_at",
            "study": "studies",
            "research": "researches",
            "analyze": "analyzes",
            "describe": "describes",
            "explain": "explains",
        }

        return relation_mappings.get(predicate.lower(), "relates_to")

    def _calculate_extraction_metadata(
        self, text: str, entities: List[Dict], relations: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate metadata about the extraction process."""
        try:
            # Entity type distribution
            entity_types = {}
            for entity in entities:
                label = entity["label"]
                entity_types[label] = entity_types.get(label, 0) + 1

            # Confidence statistics
            confidences = [entity["confidence"] for entity in entities]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            # Text characteristics
            word_count = len(text.split())
            sentence_count = text.count(".") + text.count("!") + text.count("?")

            return {
                "entity_types": entity_types,
                "total_entities": len(entities),
                "total_relations": len(relations),
                "avg_entity_confidence": avg_confidence,
                "max_confidence": max(confidences) if confidences else 0.0,
                "min_confidence": min(confidences) if confidences else 0.0,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "entity_density": len(entities) / word_count if word_count > 0 else 0.0,
            }

        except Exception as e:
            logger.error("Failed to calculate extraction metadata", error=str(e))
            return {}

    async def _process_large_text(
        self, text: str, extract_relations: bool
    ) -> Dict[str, Any]:
        """Process large text by chunking."""
        try:
            chunk_size = 5000
            overlap = 200
            chunks = []

            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i : i + chunk_size]
                chunks.append(chunk)

            # Process chunks in parallel
            chunk_tasks = [
                self.extract_entities_async(chunk, extract_relations)
                for chunk in chunks
            ]

            chunk_results = await asyncio.gather(*chunk_tasks)

            # Merge results
            all_entities = []
            all_relations = []

            for result in chunk_results:
                all_entities.extend(result.get("entities", []))
                all_relations.extend(result.get("relations", []))

            # Deduplicate merged results
            unique_entities = self._deduplicate_entities(all_entities)
            unique_relations = self._deduplicate_relations(all_relations)

            metadata = self._calculate_extraction_metadata(
                text, unique_entities, unique_relations
            )

            return {
                "entities": unique_entities,
                "relations": unique_relations,
                "metadata": metadata,
                "text_length": len(text),
                "processing_model": self.model_name,
                "chunked": True,
                "chunk_count": len(chunks),
            }

        except Exception as e:
            logger.error("Large text processing failed", error=str(e))
            return self._empty_extraction_result()

    def _deduplicate_entities(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate entities."""
        seen = set()
        unique_entities = []

        for entity in entities:
            key = (entity["text"].lower(), entity["label"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities

    def _deduplicate_relations(
        self, relations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate relations."""
        seen = set()
        unique_relations = []

        for relation in relations:
            key = (
                relation["subject"].lower(),
                relation["predicate"],
                relation["object"].lower(),
            )
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)

        return unique_relations

    def _empty_extraction_result(self) -> Dict[str, Any]:
        """Return empty extraction result."""
        return {
            "entities": [],
            "relations": [],
            "metadata": {},
            "text_length": 0,
            "processing_model": self.model_name or "none",
            "error": True,
        }

    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get statistics about the extractor."""
        return {
            "model_name": self.model_name,
            "model_available": self.nlp is not None,
            "confidence_threshold": self.confidence_threshold,
            "max_entity_length": self.max_entity_length,
            "supported_entity_types": list(self.entity_type_mapping.keys()),
        }
