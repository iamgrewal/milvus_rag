from neo4j import GraphDatabase
from graphrag.config.settings import Config
from graphrag.logger import logger

class Neo4jManager:
    def __init__(self):
        self.driver = GraphDatabase.driver(Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD))
        logger.info("Neo4j connected")

    def create_entity_and_relation(self, entity, entity_type, related, rel_type):
        if not entity or not related:
            logger.warning(f"Skipping empty entity or related: {entity}, {related}")
            return
        with self.driver.session() as session:
            session.execute_write(self._create_entity_tx, entity, entity_type, related, rel_type)

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
