# graphrag/rag_system/service.py
import asyncio
from typing import Dict, Any
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from .orchestrator import HybridRAGOrchestrator

logger = structlog.get_logger(__name__)


class RAGService:
    """
    Production-ready RAG service integrating the hybrid RAG orchestrator.

    Provides high-level interface for query processing with retry logic,
    monitoring, and error handling according to the rhoSearcher ruleset.
    """

    def __init__(self):
        """Initialize the RAG service with orchestrator and monitoring."""
        try:
            self.orchestrator = HybridRAGOrchestrator()
            self.query_count = 0
            self.error_count = 0
            self.total_processing_time = 0.0

            # Service configuration
            self.max_query_length = 10000
            self.timeout_seconds = 30
            self.enable_monitoring = True

            logger.info("RAG service initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize RAG service", error=str(e))
            raise

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def query_with_retry(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Process query with retry logic and comprehensive error handling.

        Args:
            query: User query string
            **kwargs: Additional query parameters

        Returns:
            Dictionary containing response and metadata
        """
        import time

        start_time = time.time()

        try:
            # Validate input
            validation_result = self._validate_query(query)
            if not validation_result["valid"]:
                return self._create_error_response(
                    validation_result["error"], "validation_error"
                )

            # Process query with timeout
            try:
                result = await asyncio.wait_for(
                    self._perform_rag_query(query, **kwargs),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Query processing timeout",
                    query_length=len(query),
                    timeout=self.timeout_seconds,
                )
                return self._create_error_response(
                    "Query processing timed out", "timeout_error"
                )

            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, success=True)

            # Add service metadata
            result["service_metadata"] = {
                "processing_time": processing_time,
                "service_version": "1.0.0",
                "timestamp": time.time(),
                "query_id": self.query_count,
            }

            logger.info(
                "Query processed successfully",
                query_length=len(query),
                processing_time=processing_time,
                confidence=result.get("confidence_score", 0.0),
                query_id=self.query_count,
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, success=False)

            logger.error(
                "Query processing failed",
                error=str(e),
                query_hash=hash(query),
                processing_time=processing_time,
                attempt=getattr(e, "retry_state", {}).get("attempt_number", 1),
            )

            return self._create_error_response(str(e), "processing_error")

    async def _perform_rag_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Core RAG query processing using the hybrid orchestrator.

        Args:
            query: User query string
            **kwargs: Additional parameters (confidence_threshold, max_results, etc.)

        Returns:
            Processed query result
        """
        try:
            # Extract parameters
            confidence_threshold = kwargs.get("confidence_threshold", 0.65)
            max_results = kwargs.get("max_results", 10)
            include_metadata = kwargs.get("include_metadata", True)

            # Process through orchestrator
            result = await self.orchestrator.process_query(query)

            # Apply result filtering if needed
            if max_results and "metadata" in result:
                # This is a placeholder - actual filtering would depend on result structure
                pass

            # Enhance with additional metadata if requested
            if include_metadata:
                result["enhanced_metadata"] = await self._get_enhanced_metadata(
                    query, result
                )

            return result

        except Exception as e:
            logger.error("Core RAG processing failed", error=str(e))
            raise

    async def _get_enhanced_metadata(
        self, query: str, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get enhanced metadata for the query result."""
        try:
            return {
                "query_analysis": {
                    "query_length": len(query),
                    "query_complexity": self._analyze_query_complexity(query),
                    "query_type": self._classify_query_type(query),
                },
                "result_analysis": {
                    "response_length": len(result.get("response", "")),
                    "confidence_category": self._categorize_confidence(
                        result.get("confidence_score", 0.0)
                    ),
                    "correction_applied": result.get("iteration_count", 0) > 0,
                    "sources_quality": self._assess_sources_quality(
                        result.get("metadata", {})
                    ),
                },
                "system_performance": {
                    "total_queries_processed": self.query_count,
                    "system_error_rate": self.error_count / max(self.query_count, 1),
                    "avg_processing_time": self.total_processing_time
                    / max(self.query_count, 1),
                },
            }

        except Exception as e:
            logger.error("Enhanced metadata generation failed", error=str(e))
            return {}

    def _validate_query(self, query: str) -> Dict[str, Any]:
        """Validate input query."""
        if not query or not query.strip():
            return {"valid": False, "error": "Empty query provided"}

        if len(query) > self.max_query_length:
            return {
                "valid": False,
                "error": f"Query too long (max {self.max_query_length} characters)",
            }

        # Check for potentially harmful content (basic check)
        suspicious_patterns = ["<script", "javascript:", "eval(", "exec("]
        if any(pattern in query.lower() for pattern in suspicious_patterns):
            return {"valid": False, "error": "Query contains suspicious content"}

        return {"valid": True}

    def _analyze_query_complexity(self, query: str) -> str:
        """Analyze query complexity."""
        word_count = len(query.split())
        question_words = ["what", "how", "why", "where", "when", "who", "which"]

        if word_count < 5:
            return "simple"
        elif word_count < 15:
            return "medium"
        else:
            return "complex"

    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()

        if any(word in query_lower for word in ["what is", "define", "definition"]):
            return "definition"
        elif any(word in query_lower for word in ["how to", "how do", "steps"]):
            return "instructional"
        elif any(word in query_lower for word in ["why", "reason", "because"]):
            return "explanatory"
        elif any(
            word in query_lower for word in ["compare", "difference", "vs", "versus"]
        ):
            return "comparative"
        elif any(word in query_lower for word in ["list", "examples", "types"]):
            return "enumerative"
        else:
            return "general"

    def _categorize_confidence(self, confidence: float) -> str:
        """Categorize confidence score."""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        elif confidence >= 0.4:
            return "low"
        else:
            return "very_low"

    def _assess_sources_quality(self, metadata: Dict[str, Any]) -> str:
        """Assess the quality of retrieved sources."""
        vector_count = metadata.get("vector_results_count", 0)
        graph_count = metadata.get("graph_results_count", 0)
        fused_count = metadata.get("fused_results_count", 0)

        total_sources = vector_count + graph_count

        if total_sources >= 10 and fused_count >= 5:
            return "high"
        elif total_sources >= 5 and fused_count >= 3:
            return "medium"
        elif total_sources >= 2:
            return "low"
        else:
            return "insufficient"

    def _update_metrics(self, processing_time: float, success: bool):
        """Update service metrics."""
        self.query_count += 1
        self.total_processing_time += processing_time

        if not success:
            self.error_count += 1

    def _create_error_response(
        self, error_message: str, error_type: str
    ) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "response": f"I apologize, but I encountered an error: {error_message}",
            "confidence_score": 0.0,
            "iteration_count": 0,
            "hallucination_detected": False,
            "sources_count": 0,
            "error": True,
            "error_type": error_type,
            "error_message": error_message,
            "metadata": {
                "vector_results_count": 0,
                "graph_results_count": 0,
                "fused_results_count": 0,
            },
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            health_status = {
                "service": "healthy",
                "orchestrator": "unknown",
                "vector_store": "unknown",
                "graph_store": "unknown",
                "timestamp": asyncio.get_event_loop().time(),
            }

            # Test orchestrator
            try:
                test_result = await self.orchestrator.process_query("health check test")
                health_status["orchestrator"] = (
                    "healthy" if test_result else "unhealthy"
                )
            except Exception as e:
                health_status["orchestrator"] = f"unhealthy: {str(e)[:100]}"

            # Test vector store (Milvus)
            try:
                vector_health = self.orchestrator.vector_retriever.health_check()
                health_status["vector_store"] = (
                    "healthy" if vector_health else "unhealthy"
                )
            except Exception as e:
                health_status["vector_store"] = f"unhealthy: {str(e)[:100]}"

            # Test graph store (Neo4j)
            try:
                graph_health = self.orchestrator.graph_retriever.health_check()
                health_status["graph_store"] = (
                    "healthy" if graph_health else "unhealthy"
                )
            except Exception as e:
                health_status["graph_store"] = f"unhealthy: {str(e)[:100]}"

            # Overall health assessment
            component_statuses = [
                health_status["orchestrator"],
                health_status["vector_store"],
                health_status["graph_store"],
            ]

            if all(status == "healthy" for status in component_statuses):
                health_status["overall"] = "healthy"
            elif any("healthy" in status for status in component_statuses):
                health_status["overall"] = "degraded"
            else:
                health_status["overall"] = "unhealthy"

            # Add service metrics
            health_status["metrics"] = {
                "total_queries": self.query_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.query_count, 1),
                "avg_processing_time": self.total_processing_time
                / max(self.query_count, 1),
            }

            return health_status

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                "service": "unhealthy",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time(),
            }

    def get_service_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        try:
            return {
                "service_info": {
                    "version": "1.0.0",
                    "uptime": "calculated_elsewhere",  # Would need actual uptime tracking
                    "queries_processed": self.query_count,
                    "errors_encountered": self.error_count,
                },
                "performance_metrics": {
                    "error_rate": self.error_count / max(self.query_count, 1),
                    "avg_processing_time": self.total_processing_time
                    / max(self.query_count, 1),
                    "total_processing_time": self.total_processing_time,
                },
                "configuration": {
                    "max_query_length": self.max_query_length,
                    "timeout_seconds": self.timeout_seconds,
                    "monitoring_enabled": self.enable_monitoring,
                },
            }

        except Exception as e:
            logger.error("Failed to get service statistics", error=str(e))
            return {}

    async def close(self):
        """Clean up service resources."""
        try:
            # Close orchestrator resources
            if hasattr(self.orchestrator, "close"):
                await self.orchestrator.close()

            # Close database connections
            if hasattr(self.orchestrator, "graph_retriever"):
                self.orchestrator.graph_retriever.close_connections()

            logger.info("RAG service closed successfully")

        except Exception as e:
            logger.error("Error closing RAG service", error=str(e))
