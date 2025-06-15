"""
Enhanced Neo4j Manager for Phase 2 Hybrid RAG System

Supports dynamic relationship discovery, multi-hop traversals, and integration
with the ContextEnhancementEngine for advanced graph operations.
"""

import asyncio
import time
from typing import Any, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime

from neo4j import GraphDatabase, Transaction
from neo4j.exceptions import ServiceUnavailable, TransientError

from graphrag.config.settings import Config
from graphrag.logger import logger


@dataclass
class EntityNode:
    """Represents an entity node in the graph."""
    
    name: str
    type: str
    properties: dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class RelationshipEdge:
    """Represents a relationship edge in the graph."""
    
    source: str
    target: str
    relationship_type: str
    properties: dict[str, Any]
    confidence: float
    created_at: datetime
    weight: float = 1.0


@dataclass
class GraphPath:
    """Represents a path through the graph."""
    
    nodes: list[str]
    relationships: list[str]
    length: int
    total_weight: float
    confidence: float


@dataclass
class RelationshipPattern:
    """Represents a discovered relationship pattern."""
    
    pattern_type: str
    entities: list[str]
    relationship_types: list[str]
    frequency: int
    confidence: float
    examples: list[dict[str, Any]]


class EnhancedNeo4jManager:
    """
    Enhanced Neo4j manager with advanced graph operations.
    
    Features:
    - Dynamic relationship discovery algorithms
    - Multi-hop traversal with path analysis
    - Batch operations for performance optimization
    - Integration with ContextEnhancementEngine
    - Advanced Cypher query optimization
    - Relationship pattern mining
    """
    
    def __init__(self, 
                 uri: Optional[str] = None,
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 max_connection_lifetime: int = 3600,
                 max_connection_pool_size: int = 50):
        """
        Initialize enhanced Neo4j manager.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            max_connection_lifetime: Maximum connection lifetime in seconds
            max_connection_pool_size: Maximum connection pool size
        """
        self.uri = uri or Config.NEO4J_URI
        self.user = user or Config.NEO4J_USER
        self.password = password or Config.NEO4J_PASSWORD
        
        # Initialize driver with connection pooling
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            max_connection_lifetime=max_connection_lifetime,
            max_connection_pool_size=max_connection_pool_size,
            connection_timeout=30,
            max_retry_time=15
        )
        
        # Performance tracking
        self._query_stats = defaultdict(int)
        self._query_times = defaultdict(list)
        
        logger.info(f"Enhanced Neo4j manager connected to {self.uri}")
    
    async def initialize_schema(self) -> bool:
        """
        Initialize graph schema with indexes and constraints.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                # Create constraints
                constraints = [
                    "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
                    "CREATE CONSTRAINT relationship_id_unique IF NOT EXISTS FOR ()-[r:RELATION]-() REQUIRE r.id IS UNIQUE"
                ]
                
                # Create indexes
                indexes = [
                    "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                    "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                    "CREATE INDEX relationship_type_index IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.type)",
                    "CREATE INDEX relationship_confidence_index IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.confidence)",
                    "CREATE INDEX entity_created_index IF NOT EXISTS FOR (e:Entity) ON (e.created_at)"
                ]
                
                # Execute schema creation
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        logger.debug(f"Constraint creation skipped (may already exist): {e}")
                
                for index in indexes:
                    try:
                        session.run(index)
                    except Exception as e:
                        logger.debug(f"Index creation skipped (may already exist): {e}")
                
                logger.info("Neo4j schema initialized successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j schema: {e}")
            return False
    
    async def create_entity_batch(self, entities: list[dict[str, Any]]) -> bool:
        """
        Create multiple entities in a single batch operation.
        
        Args:
            entities: List of entity dictionaries with name, type, and properties
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not entities:
                return True
            
            start_time = time.time()
            
            with self.driver.session() as session:
                # Prepare batch data
                batch_data = []
                for entity in entities:
                    batch_data.append({
                        'name': entity.get('name'),
                        'type': entity.get('type', 'UNKNOWN'),
                        'properties': entity.get('properties', {}),
                        'created_at': datetime.now().isoformat(),
                        'updated_at': datetime.now().isoformat()
                    })
                
                # Execute batch create
                result = session.run("""
                    UNWIND $entities AS entity
                    MERGE (e:Entity {name: entity.name})
                    ON CREATE SET 
                        e.type = entity.type,
                        e.created_at = entity.created_at,
                        e.updated_at = entity.updated_at,
                        e += entity.properties
                    ON MATCH SET 
                        e.updated_at = entity.updated_at,
                        e += entity.properties
                    RETURN count(e) as created_count
                """, entities=batch_data)
                
                created_count = result.single()['created_count']
                
                # Track performance
                execution_time = time.time() - start_time
                self._query_stats['create_entity_batch'] += 1
                self._query_times['create_entity_batch'].append(execution_time)
                
                logger.debug(f"Created/updated {created_count} entities in {execution_time:.3f}s")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create entity batch: {e}")
            return False
    
    async def create_relationships_batch(self, relationships: list[dict[str, Any]]) -> bool:
        """
        Create multiple relationships in a single batch operation.
        
        Args:
            relationships: List of relationship dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not relationships:
                return True
            
            start_time = time.time()
            
            with self.driver.session() as session:
                # Prepare batch data
                batch_data = []
                for rel in relationships:
                    rel_id = f"{rel['source']}_{rel['target']}_{rel['type']}_{int(time.time() * 1000)}"
                    batch_data.append({
                        'source': rel['source'],
                        'target': rel['target'],
                        'type': rel['type'],
                        'confidence': rel.get('confidence', 1.0),
                        'weight': rel.get('weight', 1.0),
                        'properties': rel.get('properties', {}),
                        'id': rel_id,
                        'created_at': datetime.now().isoformat()
                    })
                
                # Execute batch create
                result = session.run("""
                    UNWIND $relationships AS rel
                    MATCH (source:Entity {name: rel.source})
                    MATCH (target:Entity {name: rel.target})
                    MERGE (source)-[r:RELATION {type: rel.type, source: rel.source, target: rel.target}]->(target)
                    ON CREATE SET 
                        r.id = rel.id,
                        r.confidence = rel.confidence,
                        r.weight = rel.weight,
                        r.created_at = rel.created_at,
                        r += rel.properties
                    ON MATCH SET 
                        r.confidence = CASE 
                            WHEN rel.confidence > r.confidence THEN rel.confidence 
                            ELSE r.confidence 
                        END,
                        r.weight = r.weight + rel.weight,
                        r += rel.properties
                    RETURN count(r) as created_count
                """, relationships=batch_data)
                
                created_count = result.single()['created_count']
                
                # Track performance
                execution_time = time.time() - start_time
                self._query_stats['create_relationships_batch'] += 1
                self._query_times['create_relationships_batch'].append(execution_time)
                
                logger.debug(f"Created/updated {created_count} relationships in {execution_time:.3f}s")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create relationships batch: {e}")
            return False
    
    async def discover_relationship_patterns(self, 
                                           entity_types: Optional[list[str]] = None,
                                           min_frequency: int = 2,
                                           max_pattern_length: int = 3) -> list[RelationshipPattern]:
        """
        Discover common relationship patterns in the graph.
        
        Args:
            entity_types: Filter by specific entity types
            min_frequency: Minimum frequency for pattern discovery
            max_pattern_length: Maximum length of patterns to discover
            
        Returns:
            List of discovered relationship patterns
        """
        try:
            start_time = time.time()
            patterns = []
            
            with self.driver.session() as session:
                # Build entity type filter
                type_filter = ""
                if entity_types:
                    type_filter = f"WHERE e1.type IN {entity_types} AND e2.type IN {entity_types}"
                
                # Discover 2-hop patterns
                if max_pattern_length >= 2:
                    query = f"""
                        MATCH (e1:Entity)-[r1:RELATION]->(e2:Entity)-[r2:RELATION]->(e3:Entity)
                        {type_filter.replace('e2', 'e3') if type_filter else ''}
                        WITH e1.type as type1, r1.type as rel1, e2.type as type2, 
                             r2.type as rel2, e3.type as type3, 
                             count(*) as frequency
                        WHERE frequency >= $min_frequency
                        RETURN type1, rel1, type2, rel2, type3, frequency
                        ORDER BY frequency DESC
                        LIMIT 100
                    """
                    
                    result = session.run(query, min_frequency=min_frequency)
                    
                    for record in result:
                        pattern = RelationshipPattern(
                            pattern_type="2_hop",
                            entities=[record['type1'], record['type2'], record['type3']],
                            relationship_types=[record['rel1'], record['rel2']],
                            frequency=record['frequency'],
                            confidence=min(1.0, record['frequency'] / 10.0),
                            examples=[]
                        )
                        patterns.append(pattern)
                
                # Discover 3-hop patterns if requested
                if max_pattern_length >= 3:
                    query = f"""
                        MATCH (e1:Entity)-[r1:RELATION]->(e2:Entity)-[r2:RELATION]->(e3:Entity)-[r3:RELATION]->(e4:Entity)
                        {type_filter.replace('e2', 'e4') if type_filter else ''}
                        WITH e1.type as type1, r1.type as rel1, e2.type as type2, 
                             r2.type as rel2, e3.type as type3, r3.type as rel3, e4.type as type4,
                             count(*) as frequency
                        WHERE frequency >= $min_frequency
                        RETURN type1, rel1, type2, rel2, type3, rel3, type4, frequency
                        ORDER BY frequency DESC
                        LIMIT 50
                    """
                    
                    result = session.run(query, min_frequency=min_frequency)
                    
                    for record in result:
                        pattern = RelationshipPattern(
                            pattern_type="3_hop",
                            entities=[record['type1'], record['type2'], record['type3'], record['type4']],
                            relationship_types=[record['rel1'], record['rel2'], record['rel3']],
                            frequency=record['frequency'],
                            confidence=min(1.0, record['frequency'] / 5.0),
                            examples=[]
                        )
                        patterns.append(pattern)
            
            # Track performance
            execution_time = time.time() - start_time
            self._query_stats['discover_patterns'] += 1
            self._query_times['discover_patterns'].append(execution_time)
            
            logger.debug(f"Discovered {len(patterns)} relationship patterns in {execution_time:.3f}s")
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to discover relationship patterns: {e}")
            return []
    
    async def find_multi_hop_paths(self, 
                                  source: str, 
                                  target: str,
                                  max_hops: int = 3,
                                  min_confidence: float = 0.1) -> list[GraphPath]:
        """
        Find paths between entities with multi-hop traversal.
        
        Args:
            source: Source entity name
            target: Target entity name
            max_hops: Maximum number of hops
            min_confidence: Minimum confidence threshold for relationships
            
        Returns:
            List of graph paths between source and target
        """
        try:
            start_time = time.time()
            
            with self.driver.session() as session:
                query = """
                    MATCH path = (source:Entity {name: $source})-[rels:RELATION*1..%d]->(target:Entity {name: $target})
                    WHERE ALL(r IN rels WHERE r.confidence >= $min_confidence)
                    WITH path, rels,
                         reduce(weight = 0, r IN rels | weight + r.weight) as total_weight,
                         reduce(conf = 1.0, r IN rels | conf * r.confidence) as path_confidence
                    RETURN [node IN nodes(path) | node.name] as node_names,
                           [rel IN rels | rel.type] as relationship_types,
                           length(path) as path_length,
                           total_weight,
                           path_confidence
                    ORDER BY path_length, path_confidence DESC
                    LIMIT 20
                """ % max_hops
                
                result = session.run(query, 
                                   source=source, 
                                   target=target, 
                                   min_confidence=min_confidence)
                
                paths = []
                for record in result:
                    path = GraphPath(
                        nodes=record['node_names'],
                        relationships=record['relationship_types'],
                        length=record['path_length'],
                        total_weight=record['total_weight'],
                        confidence=record['path_confidence']
                    )
                    paths.append(path)
                
                # Track performance
                execution_time = time.time() - start_time
                self._query_stats['find_multi_hop_paths'] += 1
                self._query_times['find_multi_hop_paths'].append(execution_time)
                
                logger.debug(f"Found {len(paths)} paths between {source} and {target} in {execution_time:.3f}s")
                return paths
                
        except Exception as e:
            logger.error(f"Failed to find multi-hop paths: {e}")
            return []
    
    async def get_entity_neighborhood(self, 
                                    entity: str, 
                                    depth: int = 2,
                                    relationship_types: Optional[list[str]] = None) -> dict[str, Any]:
        """
        Get comprehensive neighborhood information for an entity.
        
        Args:
            entity: Entity name
            depth: Neighborhood depth
            relationship_types: Filter by specific relationship types
            
        Returns:
            Dictionary with neighborhood information
        """
        try:
            start_time = time.time()
            
            with self.driver.session() as session:
                # Build relationship type filter
                rel_filter = ""
                if relationship_types:
                    rel_filter = f"WHERE ALL(r IN rels WHERE r.type IN {relationship_types})"
                
                query = f"""
                    MATCH (center:Entity {{name: $entity}})
                    OPTIONAL MATCH path = (center)-[rels:RELATION*1..{depth}]-(neighbor:Entity)
                    {rel_filter}
                    WITH center, 
                         collect(DISTINCT neighbor.name) as neighbors,
                         collect(DISTINCT {{ 
                             neighbor: neighbor.name, 
                             relationship: [r IN rels | r.type],
                             distance: length(path),
                             confidence: reduce(conf = 1.0, r IN rels | conf * r.confidence)
                         }}) as neighbor_details
                    RETURN center.name as entity_name,
                           center.type as entity_type,
                           center.properties as entity_properties,
                           neighbors,
                           neighbor_details,
                           size(neighbors) as neighbor_count
                """
                
                result = session.run(query, entity=entity)
                record = result.single()
                
                if not record:
                    return {"entity": entity, "neighbors": [], "neighbor_count": 0}
                
                neighborhood = {
                    "entity": record['entity_name'],
                    "entity_type": record['entity_type'],
                    "entity_properties": record['entity_properties'] or {},
                    "neighbors": record['neighbors'] or [],
                    "neighbor_details": record['neighbor_details'] or [],
                    "neighbor_count": record['neighbor_count']
                }
                
                # Track performance
                execution_time = time.time() - start_time
                self._query_stats['get_neighborhood'] += 1
                self._query_times['get_neighborhood'].append(execution_time)
                
                logger.debug(f"Retrieved neighborhood for {entity} with {neighborhood['neighbor_count']} neighbors in {execution_time:.3f}s")
                return neighborhood
                
        except Exception as e:
            logger.error(f"Failed to get entity neighborhood: {e}")
            return {"entity": entity, "neighbors": [], "neighbor_count": 0}
    
    async def get_relationship_strength(self, 
                                      source: str, 
                                      target: str) -> dict[str, Any]:
        """
        Calculate relationship strength between two entities.
        
        Args:
            source: Source entity name
            target: Target entity name
            
        Returns:
            Dictionary with relationship strength metrics
        """
        try:
            with self.driver.session() as session:
                query = """
                    MATCH (s:Entity {name: $source}), (t:Entity {name: $target})
                    OPTIONAL MATCH direct_path = (s)-[direct:RELATION]-(t)
                    OPTIONAL MATCH indirect_path = (s)-[rels:RELATION*2..3]-(t)
                    WITH s, t, direct, indirect_path,
                         CASE WHEN direct IS NOT NULL THEN direct.confidence ELSE 0 END as direct_confidence,
                         CASE WHEN direct IS NOT NULL THEN direct.weight ELSE 0 END as direct_weight,
                         CASE WHEN indirect_path IS NOT NULL 
                              THEN reduce(conf = 1.0, r IN relationships(indirect_path) | conf * r.confidence) 
                              ELSE 0 END as indirect_confidence
                    RETURN s.name as source,
                           t.name as target,
                           direct_confidence,
                           direct_weight,
                           max(indirect_confidence) as max_indirect_confidence,
                           count(indirect_path) as indirect_path_count,
                           CASE WHEN direct IS NOT NULL THEN 'DIRECT' ELSE 'INDIRECT' END as connection_type
                """
                
                result = session.run(query, source=source, target=target)
                record = result.single()
                
                if not record:
                    return {"source": source, "target": target, "strength": 0.0, "connection_type": "NONE"}
                
                # Calculate overall relationship strength
                direct_conf = record['direct_confidence'] or 0
                indirect_conf = record['max_indirect_confidence'] or 0
                indirect_count = record['indirect_path_count'] or 0
                
                # Weighted combination of direct and indirect relationships
                strength = direct_conf + (indirect_conf * 0.5) + (min(indirect_count, 5) * 0.1)
                
                return {
                    "source": source,
                    "target": target,
                    "strength": min(strength, 1.0),
                    "direct_confidence": direct_conf,
                    "indirect_confidence": indirect_conf,
                    "indirect_path_count": indirect_count,
                    "connection_type": record['connection_type']
                }
                
        except Exception as e:
            logger.error(f"Failed to calculate relationship strength: {e}")
            return {"source": source, "target": target, "strength": 0.0, "connection_type": "ERROR"}
    
    async def expand_entity_context(self, 
                                  entities: list[str],
                                  expansion_factor: float = 2.0,
                                  min_confidence: float = 0.3) -> list[str]:
        """
        Expand entity context by discovering related entities.
        
        Args:
            entities: Initial entities
            expansion_factor: How many related entities to include per initial entity
            min_confidence: Minimum confidence for related entities
            
        Returns:
            Expanded list of entities
        """
        try:
            start_time = time.time()
            expanded_entities = set(entities)
            
            with self.driver.session() as session:
                query = """
                    UNWIND $entities AS entity_name
                    MATCH (e:Entity {name: entity_name})-[r:RELATION]-(related:Entity)
                    WHERE r.confidence >= $min_confidence
                    WITH related, r.confidence as confidence, count(*) as frequency
                    ORDER BY confidence DESC, frequency DESC
                    LIMIT toInteger($limit)
                    RETURN collect(related.name) as related_entities
                """
                
                limit = int(len(entities) * expansion_factor)
                result = session.run(query, 
                                   entities=entities, 
                                   min_confidence=min_confidence,
                                   limit=limit)
                
                record = result.single()
                if record and record['related_entities']:
                    expanded_entities.update(record['related_entities'])
                
                # Track performance
                execution_time = time.time() - start_time
                self._query_stats['expand_context'] += 1
                self._query_times['expand_context'].append(execution_time)
                
                logger.debug(f"Expanded {len(entities)} entities to {len(expanded_entities)} in {execution_time:.3f}s")
                return list(expanded_entities)
                
        except Exception as e:
            logger.error(f"Failed to expand entity context: {e}")
            return entities
    
    def get_performance_stats(self) -> dict[str, Any]:
        """
        Get performance statistics for Neo4j operations.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {}
        
        for operation, times in self._query_times.items():
            if times:
                stats[operation] = {
                    "total_calls": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_time": sum(times)
                }
        
        return stats
    
    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on Neo4j connection and database.
        
        Returns:
            Health check results
        """
        try:
            start_time = time.time()
            
            with self.driver.session() as session:
                # Test basic connectivity
                result = session.run("RETURN 1 as test")
                test_value = result.single()['test']
                
                # Get database info
                db_info = session.run("""
                    CALL dbms.components() YIELD name, versions, edition
                    RETURN name, versions[0] as version, edition
                """)
                
                db_record = db_info.single()
                
                # Get node and relationship counts
                count_result = session.run("""
                    MATCH (n) 
                    OPTIONAL MATCH ()-[r]->()
                    RETURN count(DISTINCT n) as node_count, count(r) as relationship_count
                """)
                
                count_record = count_result.single()
                
                response_time = time.time() - start_time
                
                return {
                    "status": "healthy",
                    "response_time": response_time,
                    "database": {
                        "name": db_record['name'],
                        "version": db_record['version'],
                        "edition": db_record['edition']
                    },
                    "statistics": {
                        "node_count": count_record['node_count'],
                        "relationship_count": count_record['relationship_count']
                    },
                    "performance": self.get_performance_stats()
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time": time.time() - start_time if 'start_time' in locals() else None
            }
    
    def close(self):
        """Close Neo4j driver connection."""
        try:
            if self.driver:
                self.driver.close()
                logger.info("Neo4j driver connection closed")
        except Exception as e:
            logger.error(f"Error closing Neo4j driver: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # Legacy methods for backward compatibility
    def create_entity_and_relation(self, entity, entity_type, related, rel_type):
        """Legacy method for backward compatibility."""
        if not entity or not related:
            logger.warning(f"Skipping empty entity or related: {entity}, {related}")
            return
        
        # Convert to new batch format
        entities = [
            {"name": entity, "type": entity_type},
            {"name": related, "type": "UNKNOWN"}
        ]
        relationships = [
            {"source": entity, "target": related, "type": rel_type}
        ]
        
        # Use async methods in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.create_entity_batch(entities))
            loop.run_until_complete(self.create_relationships_batch(relationships))
        finally:
            loop.close()
    
    def get_related(self, entity):
        """Legacy method for backward compatibility."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            neighborhood = loop.run_until_complete(self.get_entity_neighborhood(entity, depth=1))
            return neighborhood.get('neighbors', [])
        finally:
            loop.close()


# Alias for backward compatibility
Neo4jManager = EnhancedNeo4jManager