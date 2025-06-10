"""
Neo4j Graph Retriever for Hybrid RAG System

This module implements production-ready graph traversal using Neo4j with the schema design
specified in the rhoSearcher ruleset: Entity nodes with RELATES edges having strength and source.
"""

import asyncio
from typing import List, Dict, Any
from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, CypherSyntaxError
import structlog

from ..config.settings import Config

logger = structlog.get_logger(__name__)


class Neo4jRetriever:
    """
    Production-ready Neo4j graph retriever implementing the labeled property graph model
    with Entity nodes and RELATES edges with strength and source properties.
    """

    def __init__(self):
        self.uri = Config.NEO4J_URI
        self.user = Config.NEO4J_USER
        self.password = Config.NEO4J_PASSWORD
        self.driver = None
        self.async_driver = None

        # Query thresholds according to ruleset
        self.min_relationship_strength = 0.7
        self.max_traversal_depth = 3
        self.max_results_per_query = 50

        self._initialize_connections()
        self._setup_schema()

    def _initialize_connections(self):
        """Initialize both sync and async Neo4j connections."""
        try:
            # Synchronous driver for setup operations
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=30 * 60,  # 30 minutes
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,  # 60 seconds
            )

            # Asynchronous driver for retrieval operations
            self.async_driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=30 * 60,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,
            )

            # Test connections
            with self.driver.session() as session:
                session.run("RETURN 1")

            logger.info("Neo4j connections established", uri=self.uri)

        except ServiceUnavailable as e:
            logger.error("Neo4j service unavailable", error=str(e))
            raise
        except Exception as e:
            logger.error("Failed to initialize Neo4j connections", error=str(e))
            raise

    def _setup_schema(self):
        """Setup Neo4j schema with constraints and indexes."""
        try:
            with self.driver.session() as session:
                # Create constraints for Entity nodes
                session.run(
                    """
                    CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
                    FOR (e:Entity) REQUIRE e.name IS UNIQUE
                """
                )

                session.run(
                    """
                    CREATE CONSTRAINT document_id_unique IF NOT EXISTS
                    FOR (d:Document) REQUIRE d.id IS UNIQUE
                """
                )

                # Create indexes for performance
                session.run(
                    """
                    CREATE INDEX entity_type_index IF NOT EXISTS
                    FOR (e:Entity) ON (e.type)
                """
                )

                session.run(
                    """
                    CREATE INDEX entity_name_index IF NOT EXISTS
                    FOR (e:Entity) ON (e.name)
                """
                )

                session.run(
                    """
                    CREATE INDEX relationship_strength_index IF NOT EXISTS
                    FOR ()-[r:RELATES]-() ON (r.strength)
                """
                )

                session.run(
                    """
                    CREATE INDEX relationship_source_index IF NOT EXISTS
                    FOR ()-[r:RELATES]-() ON (r.source)
                """
                )

                session.run(
                    """
                    CREATE INDEX document_content_index IF NOT EXISTS
                    FOR (d:Document) ON (d.content)
                """
                )

                logger.info("Neo4j schema setup completed")

        except Exception as e:
            logger.error("Failed to setup Neo4j schema", error=str(e))
            raise

    async def traverse_async(
        self, query: str, max_depth: int = None, min_strength: float = None
    ) -> List[Dict[str, Any]]:
        """
        Asynchronous graph traversal based on query entities.

        Args:
            query: Query string to extract entities from
            max_depth: Maximum traversal depth (default from config)
            min_strength: Minimum relationship strength (default from config)

        Returns:
            List of graph traversal results with entities and relationships
        """
        if max_depth is None:
            max_depth = self.max_traversal_depth
        if min_strength is None:
            min_strength = self.min_relationship_strength

        try:
            # Extract entities from query
            query_entities = await self._extract_query_entities(query)

            if not query_entities:
                logger.warning("No entities extracted from query", query=query[:100])
                return []

            # Execute parallel graph traversals for each entity
            traversal_tasks = [
                self._traverse_from_entity(entity, max_depth, min_strength)
                for entity in query_entities
            ]

            traversal_results = await asyncio.gather(
                *traversal_tasks, return_exceptions=True
            )

            # Combine and deduplicate results
            combined_results = []
            seen_paths = set()

            for result in traversal_results:
                if isinstance(result, Exception):
                    logger.error("Traversal failed for entity", error=str(result))
                    continue

                for path_result in result:
                    path_key = self._generate_path_key(path_result)
                    if path_key not in seen_paths:
                        seen_paths.add(path_key)
                        combined_results.append(path_result)

            # Sort by relevance score
            combined_results.sort(
                key=lambda x: x.get("relevance_score", 0.0), reverse=True
            )

            logger.info(
                "Graph traversal completed",
                query_entities_count=len(query_entities),
                results_count=len(combined_results),
            )

            return combined_results[: self.max_results_per_query]

        except Exception as e:
            logger.error("Graph traversal failed", error=str(e), query=query[:100])
            return []

    async def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entities from the query string."""
        try:
            # Simple entity extraction - in production, use NLP service
            # For now, extract potential entities from query words
            words = query.lower().split()

            # Filter out common stop words and short words
            stop_words = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "what",
                "when",
                "where",
                "why",
                "how",
                "who",
                "which",
            }

            entities = [
                word for word in words if len(word) > 2 and word not in stop_words
            ]

            # Limit to top 5 entities to avoid overwhelming the graph
            return entities[:5]

        except Exception as e:
            logger.error("Entity extraction failed", error=str(e))
            return []

    async def _traverse_from_entity(
        self, entity: str, max_depth: int, min_strength: float
    ) -> List[Dict[str, Any]]:
        """Execute graph traversal from a specific entity."""
        try:
            async with self.async_driver.session() as session:
                # Main traversal query according to ruleset schema
                cypher_query = """
                MATCH path = (start:Entity {name: $entity})-[:RELATES*1..$max_depth]-(end:Entity)
                WHERE ALL(r IN relationships(path) WHERE r.strength >= $min_strength)
                WITH path, 
                     [r IN relationships(path) | r.strength] AS strengths,
                     [n IN nodes(path) | n.name] AS entity_names
                RETURN 
                    entity_names,
                    strengths,
                    length(path) as path_length,
                    reduce(total = 0.0, strength IN strengths | total + strength) / length(strengths) as avg_strength,
                    [n IN nodes(path) | {name: n.name, type: n.type, properties: properties(n)}] as entities,
                    [r IN relationships(path) | {type: type(r), strength: r.strength, source: r.source, properties: properties(r)}] as relationships
                ORDER BY avg_strength DESC, path_length ASC
                LIMIT $limit
                """

                result = await session.run(
                    cypher_query,
                    entity=entity,
                    max_depth=max_depth,
                    min_strength=min_strength,
                    limit=self.max_results_per_query // 5,  # Distribute across entities
                )

                records = await result.fetch(self.max_results_per_query // 5)

                # Process results
                processed_results = []
                for record in records:
                    path_result = {
                        "entity_names": record["entity_names"],
                        "entities": record["entities"],
                        "relationships": record["relationships"],
                        "path_length": record["path_length"],
                        "avg_strength": record["avg_strength"],
                        "relevance_score": self._calculate_relevance_score(record),
                        "source_entity": entity,
                        "retrieval_type": "graph",
                    }
                    processed_results.append(path_result)

                return processed_results

        except CypherSyntaxError as e:
            logger.error("Cypher syntax error", error=str(e), entity=entity)
            return []
        except Exception as e:
            logger.error(
                "Graph traversal failed for entity", error=str(e), entity=entity
            )
            return []

    def _calculate_relevance_score(self, record: Dict[str, Any]) -> float:
        """Calculate relevance score for a graph path."""
        try:
            avg_strength = record.get("avg_strength", 0.0)
            path_length = record.get("path_length", 1)

            # Higher strength is better, shorter paths are better
            # Score between 0 and 1
            strength_score = min(avg_strength, 1.0)
            distance_penalty = 1.0 / (1.0 + (path_length - 1) * 0.2)

            return strength_score * distance_penalty

        except Exception:
            return 0.0

    def _generate_path_key(self, path_result: Dict[str, Any]) -> str:
        """Generate a unique key for path deduplication."""
        try:
            entity_names = path_result.get("entity_names", [])
            return "_".join(sorted(entity_names))
        except Exception:
            return str(hash(str(path_result)))

    async def get_entity_context(self, entity_name: str) -> Dict[str, Any]:
        """Get detailed context for a specific entity."""
        try:
            async with self.async_driver.session() as session:
                # Query for entity details and immediate relationships
                cypher_query = """
                MATCH (e:Entity {name: $entity_name})
                OPTIONAL MATCH (e)-[r:RELATES]-(related:Entity)
                WHERE r.strength >= $min_strength
                RETURN 
                    e.name as entity_name,
                    e.type as entity_type,
                    properties(e) as entity_properties,
                    collect({
                        related_entity: related.name,
                        related_type: related.type,
                        relationship_strength: r.strength,
                        relationship_source: r.source,
                        relationship_type: type(r)
                    }) as relationships
                """

                result = await session.run(
                    cypher_query,
                    entity_name=entity_name,
                    min_strength=self.min_relationship_strength,
                )

                record = await result.single()

                if record:
                    return {
                        "entity_name": record["entity_name"],
                        "entity_type": record["entity_type"],
                        "entity_properties": record["entity_properties"],
                        "relationships": record["relationships"],
                        "relationship_count": len(record["relationships"]),
                    }
                else:
                    return {}

        except Exception as e:
            logger.error(
                "Failed to get entity context", error=str(e), entity=entity_name
            )
            return {}

    def create_entity_with_relations(self, entity_data: Dict[str, Any]) -> bool:
        """Create entities and relationships according to the schema."""
        try:
            with self.driver.session() as session:
                # Create entity
                entity_query = """
                MERGE (e:Entity {name: $name})
                SET e.type = $type,
                    e.properties = $properties,
                    e.created_at = datetime(),
                    e.updated_at = datetime()
                RETURN e
                """

                session.run(
                    entity_query,
                    name=entity_data.get("name"),
                    type=entity_data.get("type", "unknown"),
                    properties=entity_data.get("properties", {}),
                )

                # Create relationships
                relationships = entity_data.get("relationships", [])
                for rel in relationships:
                    rel_query = """
                    MATCH (e1:Entity {name: $entity1}), (e2:Entity {name: $entity2})
                    MERGE (e1)-[r:RELATES]->(e2)
                    SET r.strength = $strength,
                        r.source = $source,
                        r.type = $rel_type,
                        r.created_at = datetime()
                    """

                    session.run(
                        rel_query,
                        entity1=entity_data.get("name"),
                        entity2=rel.get("target_entity"),
                        strength=rel.get("strength", 1.0),
                        source=rel.get("source", "system"),
                        rel_type=rel.get("type", "RELATED_TO"),
                    )

                logger.info(
                    "Entity created with relationships",
                    entity=entity_data.get("name"),
                    relationships_count=len(relationships),
                )

                return True

        except Exception as e:
            logger.error("Failed to create entity", error=str(e))
            return False

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics for monitoring."""
        try:
            with self.driver.session() as session:
                # Count entities
                entity_count = session.run(
                    "MATCH (e:Entity) RETURN count(e) as count"
                ).single()["count"]

                # Count relationships
                rel_count = session.run(
                    "MATCH ()-[r:RELATES]-() RETURN count(r) as count"
                ).single()["count"]

                # Average relationship strength
                avg_strength = session.run(
                    """
                    MATCH ()-[r:RELATES]-() 
                    RETURN avg(r.strength) as avg_strength
                """
                ).single()["avg_strength"]

                # Entity types distribution
                entity_types = session.run(
                    """
                    MATCH (e:Entity) 
                    RETURN e.type as type, count(e) as count 
                    ORDER BY count DESC
                """
                ).data()

                return {
                    "entity_count": entity_count,
                    "relationship_count": rel_count,
                    "average_relationship_strength": avg_strength or 0.0,
                    "entity_types": entity_types,
                }

        except Exception as e:
            logger.error("Failed to get graph statistics", error=str(e))
            return {}

    def health_check(self) -> bool:
        """Perform health check on Neo4j connections."""
        try:
            # Test sync connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test").single()
                if result["test"] != 1:
                    return False

            # Test async connection
            async def test_async():
                async with self.async_driver.session() as session:
                    result = await session.run("RETURN 1 as test")
                    record = await result.single()
                    return record["test"] == 1

            async_result = asyncio.run(test_async())
            return async_result

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False

    def close_connections(self):
        """Clean up database connections."""
        try:
            if self.driver:
                self.driver.close()
            if self.async_driver:
                asyncio.run(self.async_driver.close())
            logger.info("Neo4j connections closed")
        except Exception as e:
            logger.error("Error closing connections", error=str(e))
