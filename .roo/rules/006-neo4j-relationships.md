---
description: "Multi-hop relationship extraction strategy for Neo4j."
globs: ["**/neo4j/**"]
alwaysApply: true
---
# Neo4j Relationship Strategy

- Extract and map relationships across:
  - Direct (sentence-level)
  - Contextual (paragraph-level)
  - Semantic (embedding-based)
- Store with Cypher `MERGE` pattern and index by `name`, `type`.

@neo4j/relationship_mapper.py
