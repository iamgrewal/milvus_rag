"""
Advanced Context Enhancement Engine for Hybrid RAG System

Implements multi-strategy relationship discovery and dynamic context graph construction
as specified in Phase 2 requirements. Provides Neo4j, semantic similarity,
co-occurrence, and temporal analysis for enhanced context discovery.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import networkx as nx
import numpy as np

from graphrag.embedding_service.service import EmbeddingService
from graphrag.logger import logger
from graphrag.neo4j.manager import Neo4jManager
from utils.model_cache import ModelCache


@dataclass
class EntityRelationship:
    """Represents a relationship between two entities with metadata."""

    source: str
    target: str
    relationship_type: str
    confidence: float
    discovery_method: str
    temporal_info: Optional[dict[str, Any]] = None
    context: Optional[str] = None


@dataclass
class ContextGraph:
    """Dynamic context graph with NetworkX integration."""

    entities: list[str]
    relationships: list[EntityRelationship]
    graph: nx.Graph
    confidence_score: float
    discovery_methods: set[str]


class ContextEnhancementEngine:
    """
    Advanced context enhancement engine implementing multi-strategy relationship discovery.

    Provides four discovery strategies:
    1. Neo4j graph traversal
    2. Semantic similarity analysis
    3. Co-occurrence pattern detection
    4. Temporal relationship analysis
    """

    def __init__(
        self,
        neo4j_manager: Optional[Neo4jManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
        similarity_threshold: float = 0.7,
        cooccurrence_window: int = 50,
        temporal_window: int = 3600,
    ):
        """
        Initialize the context enhancement engine.

        Args:
            neo4j_manager: Neo4j database manager
            embedding_service: Text embedding service
            similarity_threshold: Minimum similarity for semantic relationships
            cooccurrence_window: Word window for co-occurrence analysis
            temporal_window: Time window for temporal analysis (seconds)
        """
        self.neo4j_manager = neo4j_manager or Neo4jManager()
        self.embedding_service = embedding_service or EmbeddingService()
        self.similarity_threshold = similarity_threshold
        self.cooccurrence_window = cooccurrence_window
        self.temporal_window = temporal_window

        # Initialize model cache for semantic analysis
        self.model_cache = ModelCache()
        self.similarity_model = self.model_cache.get_embedding_model("all-MiniLM-L6-v2")

        # Entity relationship cache for performance
        self._relationship_cache: dict[str, list[EntityRelationship]] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._cache_ttl = 300  # 5 minutes

        logger.info(
            "ContextEnhancementEngine initialized with multi-strategy discovery"
        )

    async def enhance_query_context(
        self, query: str, initial_entities: list[str], max_expansion_depth: int = 2
    ) -> ContextGraph:
        """
        Enhance query context using multi-strategy relationship discovery.

        Args:
            query: Original user query
            initial_entities: Initially discovered entities
            max_expansion_depth: Maximum graph expansion depth

        Returns:
            ContextGraph with enhanced relationships and NetworkX graph
        """
        try:
            logger.info(
                f"Enhancing context for query: '{query}' with {len(initial_entities)} entities"
            )

            # Extract entities from query if none provided
            if not initial_entities:
                initial_entities = await self._extract_entities_from_query(query)

            # Parallel relationship discovery using all strategies
            relationship_tasks = [
                self._find_neo4j_relationships(initial_entities, max_expansion_depth),
                self._find_semantic_relationships(initial_entities, query),
                self._find_cooccurrence_relationships(initial_entities, query),
                self._find_temporal_relationships(initial_entities),
            ]

            all_relationships = await asyncio.gather(
                *relationship_tasks, return_exceptions=True
            )

            # Combine and deduplicate relationships
            combined_relationships = []
            discovery_methods = set()

            for i, relationships in enumerate(all_relationships):
                if isinstance(relationships, Exception):
                    logger.warning(f"Discovery method {i} failed: {relationships}")
                    continue

                combined_relationships.extend(relationships)
                discovery_methods.update(rel.discovery_method for rel in relationships)

            # Deduplicate and rank relationships
            unique_relationships = self._deduplicate_and_rank_relationships(
                combined_relationships
            )

            # Build dynamic context graph
            context_graph = self._build_context_graph(
                initial_entities, unique_relationships, discovery_methods
            )

            logger.info(
                f"Enhanced context: {len(context_graph.entities)} entities, "
                f"{len(context_graph.relationships)} relationships, "
                f"confidence: {context_graph.confidence_score:.3f}"
            )

            return context_graph

        except Exception as e:
            logger.error(f"Context enhancement failed: {e}")
            # Return minimal context graph on failure
            return ContextGraph(
                entities=initial_entities,
                relationships=[],
                graph=nx.Graph(),
                confidence_score=0.0,
                discovery_methods=set(),
            )

    async def _find_neo4j_relationships(
        self, entities: list[str], max_depth: int = 2
    ) -> list[EntityRelationship]:
        """
        Discover relationships using Neo4j graph traversal.

        Args:
            entities: Source entities for relationship discovery
            max_depth: Maximum traversal depth

        Returns:
            List of EntityRelationship objects from Neo4j
        """
        try:
            relationships = []

            # Batch query for performance optimization
            for entity in entities:
                cache_key = f"neo4j_{entity}_{max_depth}"

                # Check cache first
                if self._is_cache_valid(cache_key):
                    relationships.extend(self._relationship_cache[cache_key])
                    continue

                # Multi-hop traversal query
                with self.neo4j_manager.driver.session() as session:
                    query = f"""
                        MATCH path = (e:Entity {{name: $entity}})-[*1..{max_depth}]-(n:Entity)
                        WHERE e.name <> n.name
                        RETURN e.name as source, n.name as target,
                               relationships(path) as rels, length(path) as distance
                        ORDER BY distance LIMIT 100
                    """

                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda e=entity, q=query: session.run(q, entity=e),
                    )

                    entity_relationships = []
                    for record in result:
                        # Calculate confidence based on path distance
                        distance = record["distance"]
                        confidence = max(0.1, 1.0 - (distance - 1) * 0.3)

                        # Extract relationship type from first relationship
                        rel_type = "RELATED"
                        if record["rels"]:
                            rel_type = record["rels"][0].type

                        entity_relationships.append(
                            EntityRelationship(
                                source=record["source"],
                                target=record["target"],
                                relationship_type=rel_type,
                                confidence=confidence,
                                discovery_method="neo4j_traversal",
                                context=f"graph_distance_{distance}",
                            )
                        )

                    # Cache results
                    self._relationship_cache[cache_key] = entity_relationships
                    self._cache_timestamps[cache_key] = time.time()
                    relationships.extend(entity_relationships)

            logger.debug(f"Neo4j discovery found {len(relationships)} relationships")
            return relationships

        except Exception as e:
            logger.error(f"Neo4j relationship discovery failed: {e}")
            return []

    async def _find_semantic_relationships(
        self, entities: list[str], query: str
    ) -> list[EntityRelationship]:
        """
        Discover relationships using semantic similarity analysis.

        Args:
            entities: Source entities for similarity analysis
            query: Query context for semantic understanding

        Returns:
            List of EntityRelationship objects from semantic analysis
        """
        try:
            relationships = []

            if len(entities) < 2:
                return relationships

            # Generate embeddings for entities and query
            entity_texts = [f"{entity} in context of {query}" for entity in entities]
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None, self.similarity_model.encode, entity_texts
            )

            # Calculate pairwise similarities
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities[i + 1 :], i + 1):
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )

                    if similarity >= self.similarity_threshold:
                        relationships.append(
                            EntityRelationship(
                                source=entity1,
                                target=entity2,
                                relationship_type="SEMANTIC_SIMILAR",
                                confidence=float(similarity),
                                discovery_method="semantic_similarity",
                                context=f"similarity_{similarity:.3f}",
                            )
                        )

            logger.debug(f"Semantic discovery found {len(relationships)} relationships")
            return relationships

        except Exception as e:
            logger.error(f"Semantic relationship discovery failed: {e}")
            return []

    async def _find_cooccurrence_relationships(
        self, entities: list[str], query: str
    ) -> list[EntityRelationship]:
        """
        Discover relationships using co-occurrence pattern analysis.

        Args:
            entities: Source entities for co-occurrence analysis
            query: Query text for pattern analysis

        Returns:
            List of EntityRelationship objects from co-occurrence analysis
        """
        try:
            relationships = []

            # Tokenize query for co-occurrence analysis
            words = query.lower().split()
            if len(words) < self.cooccurrence_window:
                return relationships

            # Find entity positions in text
            entity_positions = {}
            for entity in entities:
                entity_lower = entity.lower()
                positions = []
                for i, word in enumerate(words):
                    if entity_lower in word or word in entity_lower:
                        positions.append(i)
                if positions:
                    entity_positions[entity] = positions

            # Calculate co-occurrence relationships
            for entity1, positions1 in entity_positions.items():
                for entity2, positions2 in entity_positions.items():
                    if entity1 >= entity2:  # Avoid duplicates
                        continue

                    # Find co-occurrences within window
                    cooccurrences = 0
                    for pos1 in positions1:
                        for pos2 in positions2:
                            if abs(pos1 - pos2) <= self.cooccurrence_window:
                                cooccurrences += 1

                    if cooccurrences > 0:
                        # Calculate confidence based on co-occurrence frequency
                        confidence = min(1.0, cooccurrences * 0.2)

                        relationships.append(
                            EntityRelationship(
                                source=entity1,
                                target=entity2,
                                relationship_type="CO_OCCURRENCE",
                                confidence=confidence,
                                discovery_method="cooccurrence_analysis",
                                context=f"cooccur_count_{cooccurrences}",
                            )
                        )

            logger.debug(
                f"Co-occurrence discovery found {len(relationships)} relationships"
            )
            return relationships

        except Exception as e:
            logger.error(f"Co-occurrence relationship discovery failed: {e}")
            return []

    async def _find_temporal_relationships(
        self, entities: list[str]
    ) -> list[EntityRelationship]:
        """
        Discover relationships using temporal analysis.

        Args:
            entities: Source entities for temporal analysis

        Returns:
            List of EntityRelationship objects from temporal analysis
        """
        try:
            relationships = []
            current_time = time.time()

            # Query Neo4j for temporal information
            with self.neo4j_manager.driver.session() as session:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: session.run(
                        """
                        MATCH (e1:Entity)-[r]-(e2:Entity)
                        WHERE e1.name IN $entities AND e2.name IN $entities
                        AND exists(r.timestamp)
                        RETURN e1.name as source, e2.name as target,
                               r.timestamp as timestamp, type(r) as rel_type
                        ORDER BY r.timestamp DESC
                    """,
                        entities=entities,
                    ),
                )

                # Group relationships by temporal proximity
                temporal_groups = defaultdict(list)
                for record in result:
                    timestamp = record["timestamp"]
                    if isinstance(timestamp, (int, float)):
                        time_key = int(timestamp // self.temporal_window)
                        temporal_groups[time_key].append(record)

                # Create temporal relationships
                for time_key, records in temporal_groups.items():
                    if len(records) > 1:
                        # Calculate temporal confidence based on recency and frequency
                        avg_timestamp = sum(r["timestamp"] for r in records) / len(
                            records
                        )
                        recency_factor = max(
                            0.1, 1.0 - (current_time - avg_timestamp) / (30 * 24 * 3600)
                        )
                        frequency_factor = min(1.0, len(records) * 0.1)
                        confidence = (recency_factor + frequency_factor) / 2

                        for record in records:
                            relationships.append(
                                EntityRelationship(
                                    source=record["source"],
                                    target=record["target"],
                                    relationship_type=f"TEMPORAL_{record['rel_type']}",
                                    confidence=confidence,
                                    discovery_method="temporal_analysis",
                                    temporal_info={
                                        "timestamp": record["timestamp"],
                                        "time_window": time_key,
                                        "group_size": len(records),
                                    },
                                )
                            )

            logger.debug(f"Temporal discovery found {len(relationships)} relationships")
            return relationships

        except Exception as e:
            logger.error(f"Temporal relationship discovery failed: {e}")
            return []

    def _deduplicate_and_rank_relationships(
        self, relationships: list[EntityRelationship]
    ) -> list[EntityRelationship]:
        """
        Remove duplicates and rank relationships by confidence and discovery method.

        Args:
            relationships: Raw list of relationships

        Returns:
            Deduplicated and ranked relationships
        """
        # Group by entity pair
        relationship_groups = defaultdict(list)
        for rel in relationships:
            # Normalize entity pair order
            source, target = sorted([rel.source, rel.target])
            key = f"{source}|{target}"
            relationship_groups[key].append(rel)

        # Select best relationship for each pair
        unique_relationships = []
        for _entity_pair, group in relationship_groups.items():
            # Sort by confidence and method priority
            method_priority = {
                "neo4j_traversal": 4,
                "semantic_similarity": 3,
                "temporal_analysis": 2,
                "cooccurrence_analysis": 1,
            }

            best_rel = max(
                group,
                key=lambda r: (
                    r.confidence,
                    method_priority.get(r.discovery_method, 0),
                ),
            )

            # Combine discovery methods if multiple found same relationship
            if len(group) > 1:
                methods = {rel.discovery_method for rel in group}
                best_rel.discovery_method = ",".join(sorted(methods))
                # Boost confidence for multi-method discovery
                best_rel.confidence = min(1.0, best_rel.confidence * 1.2)

            unique_relationships.append(best_rel)

        # Sort by confidence descending
        return sorted(unique_relationships, key=lambda r: r.confidence, reverse=True)

    def _build_context_graph(
        self,
        entities: list[str],
        relationships: list[EntityRelationship],
        discovery_methods: set[str],
    ) -> ContextGraph:
        """
        Build NetworkX context graph from entities and relationships.

        Args:
            entities: All discovered entities
            relationships: Discovered relationships
            discovery_methods: Methods used for discovery

        Returns:
            ContextGraph with NetworkX representation
        """
        # Create NetworkX graph
        graph = nx.Graph()

        # Add all entities as nodes
        all_entities = set(entities)
        for rel in relationships:
            all_entities.add(rel.source)
            all_entities.add(rel.target)

        for entity in all_entities:
            graph.add_node(entity, entity_type="ENTITY")

        # Add relationships as edges
        for rel in relationships:
            graph.add_edge(
                rel.source,
                rel.target,
                relationship_type=rel.relationship_type,
                confidence=rel.confidence,
                discovery_method=rel.discovery_method,
                context=rel.context,
            )

        # Calculate overall confidence score
        if relationships:
            confidence_score = sum(rel.confidence for rel in relationships) / len(
                relationships
            )
        else:
            confidence_score = 0.0

        return ContextGraph(
            entities=list(all_entities),
            relationships=relationships,
            graph=graph,
            confidence_score=confidence_score,
            discovery_methods=discovery_methods,
        )

    async def _extract_entities_from_query(self, query: str) -> list[str]:
        """
        Extract entities from query text using NLP.

        Args:
            query: Query text

        Returns:
            List of extracted entity names
        """
        try:
            # Simple entity extraction - can be enhanced with spaCy or other NLP libraries
            import re

            # Extract potential entities (capitalized words, phrases in quotes)
            entities = []

            # Capitalized words
            capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
            entities.extend(capitalized)

            # Quoted phrases
            quoted = re.findall(r'"([^"]*)"', query)
            entities.extend(quoted)

            # Remove duplicates and empty strings
            entities = list(set(filter(None, entities)))

            logger.debug(f"Extracted {len(entities)} entities from query")
            return entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid and not expired."""
        if cache_key not in self._relationship_cache:
            return False

        timestamp = self._cache_timestamps.get(cache_key, 0)
        return time.time() - timestamp < self._cache_ttl

    def get_context_summary(self, context_graph: ContextGraph) -> dict[str, Any]:
        """
        Generate summary statistics for context graph.

        Args:
            context_graph: Context graph to summarize

        Returns:
            Dictionary with summary statistics
        """
        graph = context_graph.graph

        return {
            "total_entities": len(context_graph.entities),
            "total_relationships": len(context_graph.relationships),
            "confidence_score": context_graph.confidence_score,
            "discovery_methods": list(context_graph.discovery_methods),
            "graph_density": nx.density(graph) if len(graph.nodes) > 1 else 0.0,
            "connected_components": nx.number_connected_components(graph),
            "average_clustering": (
                nx.average_clustering(graph) if len(graph.nodes) > 2 else 0.0
            ),
            "relationship_types": list(
                {rel.relationship_type for rel in context_graph.relationships}
            ),
        }
