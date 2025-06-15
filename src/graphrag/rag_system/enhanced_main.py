"""
Enhanced RAG System with Self-Correction Integration

Integrates self-correction validation at appropriate checkpoints in the
query processing pipeline for improved accuracy and reliability.
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from graphrag.config.settings import Config
from graphrag.embedding_service.service import EmbeddingService
from graphrag.graph_enrichment.context_engine import ContextEnhancementEngine
from graphrag.graph_enrichment.memory_system import ContextualMemorySystem
from graphrag.logger import logger
from graphrag.milvus.manager import MilvusManager
from graphrag.neo4j.manager import Neo4jManager
from graphrag.nlp.processor import NLPProcessor
from graphrag.validation.confidence_scoring import AdvancedConfidenceScorer
from graphrag.validation.self_correction import SelfCorrectionFramework, RAGResponse


@dataclass
class QueryContext:
    """Context information for query processing."""
    
    query: str
    preprocessed_query: str
    entities: List[Tuple[str, str]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    vector_results: List[Any] = field(default_factory=list)
    graph_context: List[str] = field(default_factory=list)
    enhanced_context: Dict[str, Any] = field(default_factory=dict)
    memory_context: Dict[str, Any] = field(default_factory=dict)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGSystemResponse:
    """Enhanced response structure with validation metadata."""
    
    answer: str
    confidence_score: float
    sources: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    correction_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class EnhancedRAGSystem:
    """
    Enhanced RAG System with integrated self-correction validation.
    
    Provides multiple validation checkpoints throughout the query processing
    pipeline to ensure high-quality, accurate responses.
    """
    
    def __init__(self):
        """Initialize the enhanced RAG system with all components."""
        try:
            # Core components
            self.milvus = MilvusManager()
            self.neo4j = Neo4jManager()
            self.nlp = NLPProcessor()
            self.embedder = EmbeddingService()
            
            # Phase 2 enhancement components
            self.context_engine = ContextEnhancementEngine()
            self.memory_system = ContextualMemorySystem()
            
            # Validation and correction components
            self.confidence_scorer = AdvancedConfidenceScorer()
            self.self_correction = SelfCorrectionFramework()
            
            # Initialize QA pipeline conditionally based on config
            self.qa_pipeline = None
            if Config.QA_PIPELINE_ENABLED:
                from transformers import pipeline
                self.qa_pipeline = pipeline(
                    "question-answering", 
                    model=Config.QA_MODEL_NAME
                )
            
            logger.info("Enhanced RAG System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced RAG System: {e}")
            raise
    
    async def ingest(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Ingest text with enhanced processing and validation.
        
        Args:
            text: Text content to ingest
            metadata: Optional metadata about the content
            
        Returns:
            Dictionary with ingestion results and statistics
        """
        start_time = time.time()
        metadata = metadata or {}
        
        try:
            logger.info(f"Starting enhanced ingestion for text length: {len(text)}")
            
            # Step 1: Extract entities and relations with validation
            entities, relations = self.nlp.extract_entities_and_relations(text)
            
            # Validation Checkpoint 1: Entity extraction quality
            entity_quality = await self._validate_entity_extraction(entities, text)
            if entity_quality["confidence"] < Config.ENTITY_EXTRACTION_MIN_CONFIDENCE:
                logger.warning(f"Low entity extraction quality: {entity_quality['confidence']}")
            
            # Step 2: Process unique entities with enhanced context
            unique_entities = {}
            processed_entities = []
            
            for entity, label in entities:
                if entity not in unique_entities:
                    unique_entities[entity] = label
                    
                    # Generate embedding
                    vector = self.embedder.embed(entity)
                    eid = int(hashlib.sha256(entity.encode()).hexdigest()[:16], 16) % (2**63)
                    
                    # Store in vector database
                    entity_data = {
                        "id": eid,
                        "entity": entity,
                        "entity_type": label,
                        "embedding": vector,
                        "source_text": text[:200],  # First 200 chars for context
                        "confidence": entity_quality.get("entity_confidences", {}).get(entity, 0.8),
                        "ingestion_timestamp": datetime.now().isoformat()
                    }
                    
                    await self.milvus.insert_async([entity_data])
                    processed_entities.append(entity)
            
            # Step 3: Process relationships with enhanced validation
            processed_relations = []
            
            for (e1, t1), (e2, t2), rel_type in relations:
                # Validation Checkpoint 2: Relationship quality
                rel_confidence = await self._validate_relationship_quality(
                    e1, e2, rel_type, text
                )
                
                if rel_confidence >= Config.RELATIONSHIP_MIN_CONFIDENCE:
                    await self.neo4j.create_entity_and_relation_async(
                        e1, t1, e2, rel_type, 
                        metadata={
                            "confidence": rel_confidence,
                            "source_text": text[:200],
                            "ingestion_timestamp": datetime.now().isoformat()
                        }
                    )
                    processed_relations.append({
                        "source": e1,
                        "target": e2,
                        "type": rel_type,
                        "confidence": rel_confidence
                    })
            
            # Step 4: Update contextual memory if enabled
            if Config.MEMORY_SYSTEM_ENABLED:
                await self.memory_system.store_context(
                    text, entities, relations, metadata
                )
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "entities_processed": len(processed_entities),
                "relationships_processed": len(processed_relations),
                "entity_quality": entity_quality,
                "processing_time": processing_time,
                "metadata": metadata
            }
            
            logger.info(f"Enhanced ingestion completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Enhanced ingestion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def answer(self, question: str, context: Dict[str, Any] = None) -> RAGSystemResponse:
        """
        Generate answer with comprehensive validation and self-correction.
        
        Args:
            question: User question
            context: Optional additional context
            
        Returns:
            Enhanced RAG response with validation results
        """
        start_time = time.time()
        context = context or {}
        
        try:
            logger.info(f"Processing question: {question}")
            
            # Initialize query context
            query_context = QueryContext(
                query=question,
                preprocessed_query=self.nlp.preprocess(question)
            )
            
            # Step 1: Preprocess and validate query
            await self._preprocess_query(query_context, context)
            
            # Validation Checkpoint 1: Query understanding
            query_validation = await self._validate_query_understanding(query_context)
            
            # Step 2: Retrieve and validate vector results
            await self._retrieve_vector_context(query_context)
            
            # Validation Checkpoint 2: Vector retrieval quality
            vector_validation = await self._validate_vector_retrieval(query_context)
            
            # Step 3: Retrieve and validate graph context
            await self._retrieve_graph_context(query_context)
            
            # Validation Checkpoint 3: Graph context quality
            graph_validation = await self._validate_graph_context(query_context)
            
            # Step 4: Enhance context with Phase 2 capabilities
            if Config.CONTEXT_ENHANCEMENT_ENABLED:
                await self._enhance_context(query_context, context)
            
            # Step 5: Retrieve memory context if enabled
            if Config.MEMORY_SYSTEM_ENABLED:
                await self._retrieve_memory_context(query_context, context)
            
            # Step 6: Generate initial response
            initial_response = await self._generate_response(query_context)
            
            # Validation Checkpoint 4: Response quality validation
            response_validation = await self._validate_response_quality(
                initial_response, query_context
            )
            
            # Step 7: Apply self-correction if enabled
            final_response = initial_response
            correction_applied = False
            
            if Config.SELF_CORRECTION_ENABLED:
                corrected_response = await self._apply_self_correction(
                    initial_response, query_context, context
                )
                
                if corrected_response != initial_response:
                    final_response = corrected_response
                    correction_applied = True
                    logger.info("Self-correction applied to response")
            
            # Step 8: Final confidence assessment
            confidence_assessment = await self.confidence_scorer.assess_confidence(
                final_response.content,
                question,
                context={
                    **context,
                    "query_context": query_context.processing_metadata
                },
                sources=final_response.sources,
                entities=final_response.entities,
                relationships=final_response.relationships
            )
            
            processing_time = time.time() - start_time
            
            # Construct comprehensive response
            system_response = RAGSystemResponse(
                answer=final_response.content,
                confidence_score=confidence_assessment.overall_confidence,
                sources=final_response.sources,
                entities=final_response.entities,
                relationships=final_response.relationships,
                validation_results=[
                    query_validation,
                    vector_validation,
                    graph_validation,
                    response_validation
                ],
                correction_suggestions=getattr(final_response, 'correction_suggestions', []),
                processing_time=processing_time,
                metadata={
                    "correction_applied": correction_applied,
                    "confidence_assessment": {
                        "metric_scores": confidence_assessment.metric_scores,
                        "uncertainty_indicators": len(confidence_assessment.uncertainty_indicators),
                        "validation_passed": len([vr for vr in confidence_assessment.validation_results if vr.get("passed", False)])
                    },
                    "processing_checkpoints": {
                        "query_preprocessing": "completed",
                        "vector_retrieval": "completed",
                        "graph_retrieval": "completed",
                        "context_enhancement": "completed" if Config.CONTEXT_ENHANCEMENT_ENABLED else "skipped",
                        "memory_retrieval": "completed" if Config.MEMORY_SYSTEM_ENABLED else "skipped",
                        "self_correction": "applied" if correction_applied else "not_needed"
                    },
                    "query_context_metadata": query_context.processing_metadata
                }
            )
            
            logger.info(f"Question answered in {processing_time:.2f}s with confidence {confidence_assessment.overall_confidence:.3f}")
            return system_response
            
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return RAGSystemResponse(
                answer="I apologize, but I encountered an error while processing your question.",
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    async def _preprocess_query(self, query_context: QueryContext, context: Dict[str, Any]) -> None:
        """Preprocess query and extract initial entities."""
        try:
            # Extract entities from the query
            entities, relations = self.nlp.extract_entities_and_relations(query_context.query)
            query_context.entities = entities
            query_context.relationships = [
                {"source": e1[0], "target": e2[0], "type": rel_type}
                for (e1, _), (e2, _), rel_type in relations
            ]
            
            query_context.processing_metadata["preprocessing"] = {
                "entities_found": len(entities),
                "relationships_found": len(relations),
                "query_length": len(query_context.query)
            }
            
        except Exception as e:
            logger.error(f"Query preprocessing failed: {e}")
            query_context.processing_metadata["preprocessing_error"] = str(e)
    
    async def _validate_query_understanding(self, query_context: QueryContext) -> Dict[str, Any]:
        """Validate query understanding and entity extraction."""
        try:
            confidence = 0.8  # Base confidence
            
            # Check if entities were extracted
            if query_context.entities:
                confidence += 0.1
            
            # Check query complexity
            if len(query_context.query.split()) > 5:
                confidence += 0.05
            
            # Check for question words
            question_words = ["what", "who", "where", "when", "why", "how"]
            if any(word in query_context.query.lower() for word in question_words):
                confidence += 0.05
            
            return {
                "checkpoint": "query_understanding",
                "passed": confidence >= 0.7,
                "confidence": min(confidence, 1.0),
                "details": {
                    "entities_extracted": len(query_context.entities),
                    "has_question_words": any(word in query_context.query.lower() for word in question_words)
                }
            }
            
        except Exception as e:
            logger.error(f"Query validation failed: {e}")
            return {
                "checkpoint": "query_understanding",
                "passed": False,
                "confidence": 0.5,
                "error": str(e)
            }
    
    async def _retrieve_vector_context(self, query_context: QueryContext) -> None:
        """Retrieve vector context from Milvus."""
        try:
            vector = self.embedder.embed(query_context.preprocessed_query)
            results = await self.milvus.search_async(vector, top_k=Config.VECTOR_SEARCH_TOP_K)
            query_context.vector_results = results
            
            query_context.processing_metadata["vector_retrieval"] = {
                "results_count": len(results) if results else 0,
                "top_score": results[0].distance if results else 0.0
            }
            
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            query_context.processing_metadata["vector_retrieval_error"] = str(e)
    
    async def _validate_vector_retrieval(self, query_context: QueryContext) -> Dict[str, Any]:
        """Validate vector retrieval quality."""
        try:
            if not query_context.vector_results:
                return {
                    "checkpoint": "vector_retrieval",
                    "passed": False,
                    "confidence": 0.0,
                    "details": {"message": "No vector results found"}
                }
            
            # Calculate confidence based on result quality
            top_score = query_context.vector_results[0].distance if query_context.vector_results else 0.0
            confidence = min(top_score * 2, 1.0)  # Normalize distance to confidence
            
            return {
                "checkpoint": "vector_retrieval",
                "passed": confidence >= Config.VECTOR_SIMILARITY_THRESHOLD,
                "confidence": confidence,
                "details": {
                    "results_count": len(query_context.vector_results),
                    "top_score": top_score
                }
            }
            
        except Exception as e:
            logger.error(f"Vector validation failed: {e}")
            return {
                "checkpoint": "vector_retrieval",
                "passed": False,
                "confidence": 0.5,
                "error": str(e)
            }
    
    async def _retrieve_graph_context(self, query_context: QueryContext) -> None:
        """Retrieve graph context from Neo4j."""
        try:
            entities = [entity for entity, _ in query_context.entities]
            
            if query_context.vector_results:
                # Add entities from vector results
                vector_entities = [r.entity.get("entity") for r in query_context.vector_results[:5]]
                entities.extend(vector_entities)
            
            # Remove duplicates
            entities = list(set(entities))
            
            # Retrieve related entities
            context = []
            for entity in entities:
                related = await self.neo4j.get_related_async(entity)
                context.extend(related)
            
            query_context.graph_context = list(set(context))
            
            query_context.processing_metadata["graph_retrieval"] = {
                "query_entities": len(entities),
                "related_entities": len(query_context.graph_context)
            }
            
        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            query_context.processing_metadata["graph_retrieval_error"] = str(e)
    
    async def _validate_graph_context(self, query_context: QueryContext) -> Dict[str, Any]:
        """Validate graph context quality."""
        try:
            context_count = len(query_context.graph_context)
            
            # Calculate confidence based on context richness
            if context_count == 0:
                confidence = 0.1
            elif context_count < 3:
                confidence = 0.5
            elif context_count < 10:
                confidence = 0.8
            else:
                confidence = 0.9
            
            return {
                "checkpoint": "graph_context",
                "passed": context_count > 0,
                "confidence": confidence,
                "details": {
                    "context_entities": context_count,
                    "has_sufficient_context": context_count >= 3
                }
            }
            
        except Exception as e:
            logger.error(f"Graph validation failed: {e}")
            return {
                "checkpoint": "graph_context",
                "passed": False,
                "confidence": 0.5,
                "error": str(e)
            }
    
    async def _enhance_context(self, query_context: QueryContext, context: Dict[str, Any]) -> None:
        """Enhance context using Phase 2 context enhancement engine."""
        try:
            enhanced_context = await self.context_engine.enhance_query_context(
                query_context.query,
                query_context.entities,
                additional_context=context
            )
            
            query_context.enhanced_context = enhanced_context
            
            query_context.processing_metadata["context_enhancement"] = {
                "relationships_discovered": len(enhanced_context.get("discovered_relationships", [])),
                "context_expansion": enhanced_context.get("expansion_factor", 1.0)
            }
            
        except Exception as e:
            logger.error(f"Context enhancement failed: {e}")
            query_context.processing_metadata["context_enhancement_error"] = str(e)
    
    async def _retrieve_memory_context(self, query_context: QueryContext, context: Dict[str, Any]) -> None:
        """Retrieve context from memory system."""
        try:
            memory_context = await self.memory_system.retrieve_similar_contexts(
                query_context.query,
                limit=Config.MEMORY_RETRIEVAL_LIMIT
            )
            
            query_context.memory_context = memory_context
            
            query_context.processing_metadata["memory_retrieval"] = {
                "similar_contexts": len(memory_context.get("contexts", [])),
                "relevance_score": memory_context.get("average_similarity", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            query_context.processing_metadata["memory_retrieval_error"] = str(e)
    
    async def _generate_response(self, query_context: QueryContext) -> RAGResponse:
        """Generate initial response from context."""
        try:
            # Combine all context sources
            context_parts = []
            
            # Add graph context
            if query_context.graph_context:
                context_parts.append(". ".join([f"{e} is related" for e in query_context.graph_context[:10]]))
            
            # Add enhanced context
            if query_context.enhanced_context.get("enhanced_context"):
                context_parts.append(query_context.enhanced_context["enhanced_context"])
            
            # Add memory context
            if query_context.memory_context.get("contexts"):
                memory_text = " ".join([ctx.get("text", "") for ctx in query_context.memory_context["contexts"][:3]])
                context_parts.append(memory_text)
            
            context_text = ". ".join(context_parts) if context_parts else "No relevant context found."
            
            # Generate answer using QA pipeline or fallback
            if self.qa_pipeline and context_text.strip():
                try:
                    answer = self.qa_pipeline(
                        question=query_context.query, 
                        context=context_text
                    )['answer']
                except Exception as qa_error:
                    logger.warning(f"QA pipeline failed, using fallback: {qa_error}")
                    answer = self._generate_fallback_answer(query_context, context_text)
            else:
                answer = self._generate_fallback_answer(query_context, context_text)
            
            # Extract entities and relationships from the response
            response_entities = [entity for entity, _ in query_context.entities]
            response_relationships = query_context.relationships
            
            # Create sources list
            sources = []
            if query_context.vector_results:
                for result in query_context.vector_results[:3]:
                    sources.append({
                        "type": "vector_search",
                        "entity": result.entity.get("entity", ""),
                        "confidence": result.distance,
                        "content": result.entity.get("source_text", "")
                    })
            
            if query_context.memory_context.get("contexts"):
                for ctx in query_context.memory_context["contexts"][:2]:
                    sources.append({
                        "type": "memory_context",
                        "content": ctx.get("text", ""),
                        "confidence": ctx.get("similarity", 0.0)
                    })
            
            return RAGResponse(
                content=answer,
                sources=sources,
                entities=response_entities,
                relationships=response_relationships,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return RAGResponse(
                content="I apologize, but I couldn't generate a response based on the available information.",
                sources=[],
                entities=[],
                relationships=[],
                timestamp=datetime.now()
            )
    
    def _generate_fallback_answer(self, query_context: QueryContext, context_text: str) -> str:
        """Generate fallback answer when QA pipeline is unavailable."""
        if not context_text or context_text == "No relevant context found.":
            return "I don't have sufficient information to answer your question."
        
        # Simple template-based response
        if query_context.entities:
            entity_names = [entity for entity, _ in query_context.entities]
            return f"Based on the available information about {', '.join(entity_names)}, I found relevant context but cannot provide a detailed answer without advanced processing capabilities."
        
        return "I found some relevant information but cannot provide a detailed answer at this time."
    
    async def _validate_response_quality(self, response: RAGResponse, query_context: QueryContext) -> Dict[str, Any]:
        """Validate response quality before self-correction."""
        try:
            # Basic quality checks
            response_length = len(response.content)
            has_entities = len(response.entities) > 0
            has_sources = len(response.sources) > 0
            
            # Calculate confidence
            confidence = 0.5  # Base confidence
            
            if response_length > 10:
                confidence += 0.2
            if has_entities:
                confidence += 0.15
            if has_sources:
                confidence += 0.15
            
            # Check for obvious issues
            issues = []
            if response_length < 5:
                issues.append("Response too short")
            if "apologize" in response.content.lower():
                issues.append("Response contains apology")
            
            return {
                "checkpoint": "response_quality",
                "passed": len(issues) == 0 and confidence >= 0.6,
                "confidence": min(confidence, 1.0),
                "details": {
                    "response_length": response_length,
                    "has_entities": has_entities,
                    "has_sources": has_sources,
                    "issues": issues
                }
            }
            
        except Exception as e:
            logger.error(f"Response validation failed: {e}")
            return {
                "checkpoint": "response_quality",
                "passed": False,
                "confidence": 0.5,
                "error": str(e)
            }
    
    async def _apply_self_correction(
        self, 
        response: RAGResponse, 
        query_context: QueryContext, 
        context: Dict[str, Any]
    ) -> RAGResponse:
        """Apply self-correction to the response."""
        try:
            # Prepare context for self-correction
            correction_context = {
                **context,
                "query_entities": query_context.entities,
                "vector_results": query_context.processing_metadata.get("vector_retrieval", {}),
                "graph_context": query_context.processing_metadata.get("graph_retrieval", {}),
                "processing_metadata": query_context.processing_metadata
            }
            
            # Apply self-correction
            corrected_response = await self.self_correction.validate_and_correct(
                response,
                query_context.query,
                correction_context
            )
            
            return corrected_response
            
        except Exception as e:
            logger.error(f"Self-correction failed: {e}")
            return response  # Return original response if correction fails
    
    async def _validate_entity_extraction(self, entities: List[Tuple[str, str]], text: str) -> Dict[str, Any]:
        """Validate entity extraction quality."""
        try:
            entity_count = len(entities)
            text_length = len(text.split())
            
            # Calculate entity density
            entity_density = entity_count / text_length if text_length > 0 else 0
            
            # Determine quality based on density and count
            if entity_density > 0.3:  # Too many entities
                confidence = 0.6
            elif entity_density < 0.05:  # Too few entities
                confidence = 0.7
            else:
                confidence = 0.9
            
            # Individual entity confidence (simplified)
            entity_confidences = {entity: 0.8 for entity, _ in entities}
            
            return {
                "confidence": confidence,
                "entity_count": entity_count,
                "entity_density": entity_density,
                "entity_confidences": entity_confidences
            }
            
        except Exception as e:
            logger.error(f"Entity validation failed: {e}")
            return {"confidence": 0.5, "error": str(e)}
    
    async def _validate_relationship_quality(
        self, 
        entity1: str, 
        entity2: str, 
        relation_type: str, 
        text: str
    ) -> float:
        """Validate relationship quality."""
        try:
            # Check if both entities appear in the text
            text_lower = text.lower()
            e1_in_text = entity1.lower() in text_lower
            e2_in_text = entity2.lower() in text_lower
            
            if not (e1_in_text and e2_in_text):
                return 0.3
            
            # Check proximity of entities in text
            e1_pos = text_lower.find(entity1.lower())
            e2_pos = text_lower.find(entity2.lower())
            
            if e1_pos >= 0 and e2_pos >= 0:
                distance = abs(e1_pos - e2_pos)
                # Closer entities have higher confidence
                proximity_confidence = max(0.5, 1.0 - (distance / len(text)))
            else:
                proximity_confidence = 0.5
            
            # Base confidence for having both entities
            base_confidence = 0.7
            
            return min(base_confidence + (proximity_confidence * 0.3), 1.0)
            
        except Exception as e:
            logger.error(f"Relationship validation failed: {e}")
            return 0.5
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of all system components."""
        try:
            start_time = time.time()
            
            # Check core components
            components_health = {}
            
            # Milvus health
            try:
                milvus_health = await self.milvus.health_check()
                components_health["milvus"] = milvus_health
            except Exception as e:
                components_health["milvus"] = {"status": "unhealthy", "error": str(e)}
            
            # Neo4j health
            try:
                neo4j_health = await self.neo4j.health_check()
                components_health["neo4j"] = neo4j_health
            except Exception as e:
                components_health["neo4j"] = {"status": "unhealthy", "error": str(e)}
            
            # Context enhancement health
            if Config.CONTEXT_ENHANCEMENT_ENABLED:
                try:
                    context_health = await self.context_engine.health_check()
                    components_health["context_engine"] = context_health
                except Exception as e:
                    components_health["context_engine"] = {"status": "unhealthy", "error": str(e)}
            
            # Memory system health
            if Config.MEMORY_SYSTEM_ENABLED:
                try:
                    memory_health = await self.memory_system.health_check()
                    components_health["memory_system"] = memory_health
                except Exception as e:
                    components_health["memory_system"] = {"status": "unhealthy", "error": str(e)}
            
            # Self-correction health
            if Config.SELF_CORRECTION_ENABLED:
                try:
                    correction_health = await self.self_correction.health_check()
                    components_health["self_correction"] = correction_health
                except Exception as e:
                    components_health["self_correction"] = {"status": "unhealthy", "error": str(e)}
            
            # Confidence scorer health
            try:
                confidence_health = await self.confidence_scorer.health_check()
                components_health["confidence_scorer"] = confidence_health
            except Exception as e:
                components_health["confidence_scorer"] = {"status": "unhealthy", "error": str(e)}
            
            # Overall health assessment
            healthy_components = [
                comp for comp in components_health.values()
                if comp.get("status") == "healthy"
            ]
            
            total_components = len(components_health)
            healthy_count = len(healthy_components)
            health_ratio = healthy_count / total_components if total_components > 0 else 0
            
            overall_status = "healthy" if health_ratio >= 0.8 else "degraded" if health_ratio >= 0.5 else "unhealthy"
            
            response_time = time.time() - start_time
            
            return {
                "status": overall_status,
                "response_time": response_time,
                "components": components_health,
                "health_summary": {
                    "total_components": total_components,
                    "healthy_components": healthy_count,
                    "health_ratio": health_ratio
                },
                "configuration": {
                    "self_correction_enabled": Config.SELF_CORRECTION_ENABLED,
                    "context_enhancement_enabled": Config.CONTEXT_ENHANCEMENT_ENABLED,
                    "memory_system_enabled": Config.MEMORY_SYSTEM_ENABLED,
                    "qa_pipeline_enabled": Config.QA_PIPELINE_ENABLED
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time": time.time() - start_time
            }


# Convenience function for backward compatibility
async def create_rag_system() -> EnhancedRAGSystem:
    """Create and initialize enhanced RAG system."""
    return EnhancedRAGSystem()


if __name__ == "__main__":
    async def main():
        """Main execution function for testing."""
        rag = EnhancedRAGSystem()
        
        # Test document
        doc = """
        Apple Inc. was founded by Steve Jobs in 1976. Steve Jobs was succeeded by Tim Cook as CEO in 2011.
        Tim Cook has been leading Apple for over a decade. Apple is headquartered in Cupertino, California.
        California is a US state known for its technology companies.
        """
        
        print("🔄 Ingesting document...")
        ingestion_result = await rag.ingest(doc)
        print(f"✅ Ingestion completed: {ingestion_result}")
        
        # Test question
        question = "Who is the current CEO of Apple?"
        print(f"\n❓ Question: {question}")
        
        response = await rag.answer(question)
        print(f"🤖 Answer: {response.answer}")
        print(f"📊 Confidence: {response.confidence_score:.3f}")
        print(f"⏱️  Processing time: {response.processing_time:.2f}s")
        print(f"🔍 Validation results: {len(response.validation_results)} checkpoints")
        
        # Print validation details
        for i, validation in enumerate(response.validation_results, 1):
            status = "✅" if validation.get("passed", False) else "❌"
            print(f"  {status} Checkpoint {i}: {validation.get('checkpoint', 'unknown')} "
                  f"(confidence: {validation.get('confidence', 0):.3f})")
        
        # Health check
        print("\n🏥 Health Check:")
        health = await rag.health_check()
        print(f"Overall Status: {health['status']}")
        print(f"Components: {health['health_summary']['healthy_components']}/{health['health_summary']['total_components']} healthy")
    
    # Run the main function
    asyncio.run(main())