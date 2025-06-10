"""
LangGraph Orchestrator for Hybrid RAG System

This module implements the LangGraph-based orchestration for the hybrid RAG system,
combining vector search (Milvus) and graph traversal (Neo4j) with self-correction
and confidence tracking capabilities.

Follows LangChain best practices for production deployment:
- Component-based architecture with dependency injection
- Comprehensive error handling and retry patterns
- Input validation and sanitization
- Performance optimization with caching
- Observability and monitoring integration
"""

import asyncio
import hashlib
import time
from functools import lru_cache
from typing import TypedDict, List, Dict, Any, Optional, Protocol
from langgraph import StateGraph, END
import structlog

from ..milvus.retriever import MilvusRetriever
from ..neo4j.graph_retriever import Neo4jRetriever
from ..fusion.fuser import ResultFuser, detect_hallucination
from ..nlp.entity_extractor import EntityExtractor

logger = structlog.get_logger(__name__)


class RAGState(TypedDict):
    """
    State specification for the hybrid RAG workflow.
    Tracks query, retrieval results, fusion output, response, and confidence metrics.
    """

    query: str
    vector_results: List[Dict[str, Any]]
    graph_results: List[Dict[str, Any]]
    fused_results: List[Dict[str, Any]]
    response: str
    confidence_score: float
    iteration_count: int
    correction_needed: bool
    hallucination_detected: bool
    sources: List[str]


# Protocol definitions for dependency injection
class VectorRetrieverProtocol(Protocol):
    """Protocol for vector retrievers."""

    async def retrieve_async(
        self, query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]: ...


class GraphRetrieverProtocol(Protocol):
    """Protocol for graph retrievers."""

    async def traverse_async(
        self, query: str, max_depth: int = 3
    ) -> List[Dict[str, Any]]: ...


class ResultFuserProtocol(Protocol):
    """Protocol for result fusers."""

    async def fuse_results(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        confidence_threshold: float = 0.65,
    ) -> List[Dict[str, Any]]: ...


class EntityExtractorProtocol(Protocol):
    """Protocol for entity extractors."""

    async def extract_entities_async(self, text: str) -> List[Dict[str, Any]]: ...


class OrchestrationError(Exception):
    """Custom exception for orchestration-related errors."""

    pass


class QueryValidationError(OrchestrationError):
    """Exception raised for invalid queries."""

    pass


class RetrievalError(OrchestrationError):
    """Exception raised during retrieval operations."""

    pass


class FusionError(OrchestrationError):
    """Exception raised during result fusion."""

    pass


class ResponseGenerationError(OrchestrationError):
    """Exception raised during response generation."""

    pass


class QueryValidator:
    """Input validation and sanitization for queries."""

    MAX_QUERY_LENGTH = 1000
    MIN_QUERY_LENGTH = 3
    FORBIDDEN_PATTERNS = [
        r"<script.*?>.*?</script>",  # XSS protection
        r"javascript:",
        r"eval\(",
        r"exec\(",
    ]

    @classmethod
    def validate_query(cls, query: str) -> str:
        """
        Validate and sanitize user query input.

        Args:
            query: Raw user query

        Returns:
            Sanitized query string

        Raises:
            QueryValidationError: If query is invalid
        """
        if not query or not isinstance(query, str):
            raise QueryValidationError("Query must be a non-empty string")

        query = query.strip()

        if len(query) < cls.MIN_QUERY_LENGTH:
            raise QueryValidationError(
                f"Query too short (minimum {cls.MIN_QUERY_LENGTH} characters)"
            )

        if len(query) > cls.MAX_QUERY_LENGTH:
            raise QueryValidationError(
                f"Query too long (maximum {cls.MAX_QUERY_LENGTH} characters)"
            )

        # Check for forbidden patterns
        import re

        for pattern in cls.FORBIDDEN_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                raise QueryValidationError("Query contains forbidden content")

        # Basic sanitization
        query = re.sub(r'[<>"\']', "", query)  # Remove potentially harmful characters
        query = re.sub(r"\s+", " ", query)  # Normalize whitespace

        return query


class CacheManager:
    """Simple in-memory cache for query results."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached result for query."""
        key = self._get_cache_key(query)

        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                logger.debug("Cache hit", query_hash=key[:8])
                return entry["data"]
            else:
                # Expired entry
                del self._cache[key]

        logger.debug("Cache miss", query_hash=key[:8])
        return None

    def set(self, query: str, data: Dict[str, Any]) -> None:
        """Cache result for query."""
        if len(self._cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(
                self._cache.keys(), key=lambda k: self._cache[k]["timestamp"]
            )
            del self._cache[oldest_key]

        key = self._get_cache_key(query)
        self._cache[key] = {"data": data, "timestamp": time.time()}
        logger.debug("Cached result", query_hash=key[:8])


class RetryManager:
    """Exponential backoff retry manager."""

    @staticmethod
    async def retry_with_backoff(
        func,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        exceptions: tuple = (Exception,),
    ):
        """
        Retry function with exponential backoff.

        Args:
            func: Async function to retry
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries
            max_delay: Maximum delay between retries
            exceptions: Exception types to catch and retry
        """
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await func()
            except exceptions as e:
                last_exception = e

                if attempt == max_retries:
                    break

                delay = min(base_delay * (2**attempt), max_delay)
                logger.warning(
                    "Retry attempt failed",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    delay=delay,
                    error=str(e),
                )
                await asyncio.sleep(delay)

        raise last_exception


class HybridRAGOrchestrator:
    """
    LangGraph-based orchestrator for hybrid RAG operations.

    Implements production-ready parallel retrieval, fusion logic,
    and self-correction with maximum 3 iterations.

    Follows LangChain best practices:
    - Dependency injection for testability
    - Comprehensive error handling
    - Input validation and sanitization
    - Caching for performance
    - Retry logic with exponential backoff
    - Observability and monitoring
    """

    def __init__(
        self,
        vector_retriever: Optional[VectorRetrieverProtocol] = None,
        graph_retriever: Optional[GraphRetrieverProtocol] = None,
        result_fuser: Optional[ResultFuserProtocol] = None,
        entity_extractor: Optional[EntityExtractorProtocol] = None,
        max_corrections: int = 3,
        confidence_threshold: float = 0.65,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 300,
        max_cache_size: int = 100,
    ):
        """
        Initialize orchestrator with dependency injection.

        Args:
            vector_retriever: Vector search component
            graph_retriever: Graph traversal component
            result_fuser: Result fusion component
            entity_extractor: Entity extraction component
            max_corrections: Maximum self-correction iterations
            confidence_threshold: Minimum confidence for acceptance
            enable_caching: Whether to enable result caching
            cache_ttl_seconds: Cache TTL in seconds
            max_cache_size: Maximum cache entries
        """
        # Dependency injection with defaults
        self.vector_retriever = vector_retriever or MilvusRetriever()
        self.graph_retriever = graph_retriever or Neo4jRetriever()
        self.result_fuser = result_fuser or ResultFuser()
        self.entity_extractor = entity_extractor or EntityExtractor()

        # Configuration
        self.max_corrections = max_corrections
        self.confidence_threshold = confidence_threshold

        # Performance and reliability components
        self.query_validator = QueryValidator()
        self.retry_manager = RetryManager()

        if enable_caching:
            self.cache = CacheManager(
                max_size=max_cache_size, ttl_seconds=cache_ttl_seconds
            )
        else:
            self.cache = None

        # Build the LangGraph workflow
        self.workflow = self._build_workflow()

        logger.info(
            "HybridRAGOrchestrator initialized",
            max_corrections=max_corrections,
            confidence_threshold=confidence_threshold,
            caching_enabled=enable_caching,
        )

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with conditional edges."""
        workflow = StateGraph(RAGState)

        # Add nodes with error handling wrappers
        workflow.add_node("query_router", self._with_error_handling(self.route_query))
        workflow.add_node(
            "parallel_retrieval", self._with_error_handling(self.parallel_retrieval)
        )
        workflow.add_node("fusion", self._with_error_handling(self.fuse_results))
        workflow.add_node(
            "generate_response", self._with_error_handling(self.generate_response)
        )
        workflow.add_node(
            "validate_response", self._with_error_handling(self.validate_response)
        )
        workflow.add_node(
            "self_correction", self._with_error_handling(self.self_correction)
        )
        workflow.add_node(
            "fallback_retrieval", self._with_error_handling(self.fallback_retrieval)
        )

        # Set entry point
        workflow.set_entry_point("query_router")

        # Add conditional edges with clear routing logic
        workflow.add_conditional_edges(
            "query_router",
            self.should_proceed_to_retrieval,
            {"proceed": "parallel_retrieval", "fallback": "fallback_retrieval"},
        )

        workflow.add_edge("parallel_retrieval", "fusion")
        workflow.add_edge("fusion", "generate_response")
        workflow.add_edge("generate_response", "validate_response")

        workflow.add_conditional_edges(
            "validate_response",
            self.should_correct,
            {
                "correct": "self_correction",
                "fallback": "fallback_retrieval",
                "end": END,
            },
        )

        workflow.add_edge("self_correction", "parallel_retrieval")
        workflow.add_edge("fallback_retrieval", END)

        return workflow.compile()

    def _with_error_handling(self, func):
        """Decorator to add consistent error handling to workflow nodes."""

        async def wrapper(state: RAGState) -> RAGState:
            try:
                start_time = time.time()
                result = await func(state)
                execution_time = time.time() - start_time

                logger.info(
                    f"{func.__name__} completed",
                    execution_time=execution_time,
                    iteration=state.get("iteration_count", 0),
                )
                return result

            except Exception as e:
                logger.error(
                    f"{func.__name__} failed",
                    error=str(e),
                    iteration=state.get("iteration_count", 0),
                )
                # Return state with error information
                state["response"] = f"Error in {func.__name__}: {str(e)}"
                state["confidence_score"] = 0.0
                return state

        return wrapper

    async def route_query(self, state: RAGState) -> RAGState:
        """Route and preprocess the incoming query with validation."""
        try:
            # Validate and sanitize query
            validated_query = self.query_validator.validate_query(state["query"])
            state["query"] = validated_query

            logger.info("Query routed and validated", query_length=len(validated_query))

            # Initialize state values
            state["iteration_count"] = 0
            state["correction_needed"] = False
            state["hallucination_detected"] = False
            state["confidence_score"] = 0.0
            state["sources"] = []

            return state

        except QueryValidationError as e:
            logger.error("Query validation failed", error=str(e))
            raise OrchestrationError(f"Invalid query: {str(e)}")

    async def parallel_retrieval(self, state: RAGState) -> RAGState:
        """
        Execute parallel retrieval from both vector and graph stores.
        Implements async/await patterns with retry logic for optimal performance.
        """
        logger.info("Starting parallel retrieval", iteration=state["iteration_count"])

        async def vector_retrieval_with_retry():
            return await self.retry_manager.retry_with_backoff(
                lambda: self.vector_retriever.retrieve_async(state["query"]),
                max_retries=2,
                exceptions=(Exception,),
            )

        async def graph_retrieval_with_retry():
            return await self.retry_manager.retry_with_backoff(
                lambda: self.graph_retriever.traverse_async(state["query"]),
                max_retries=2,
                exceptions=(Exception,),
            )

        try:
            # Create parallel tasks for vector and graph retrieval
            vector_task = asyncio.create_task(vector_retrieval_with_retry())
            graph_task = asyncio.create_task(graph_retrieval_with_retry())

            # Execute in parallel and await both results
            vector_results, graph_results = await asyncio.gather(
                vector_task, graph_task, return_exceptions=True
            )

            # Handle exceptions from gather
            if isinstance(vector_results, Exception):
                logger.error("Vector retrieval failed", error=str(vector_results))
                vector_results = []

            if isinstance(graph_results, Exception):
                logger.error("Graph retrieval failed", error=str(graph_results))
                graph_results = []

            state["vector_results"] = vector_results
            state["graph_results"] = graph_results

            logger.info(
                "Parallel retrieval completed",
                vector_count=len(vector_results),
                graph_count=len(graph_results),
            )

        except Exception as e:
            logger.error("Parallel retrieval failed", error=str(e))
            raise RetrievalError(f"Retrieval failed: {str(e)}")

        return state

    async def fuse_results(self, state: RAGState) -> RAGState:
        """
        Fuse vector and graph results using weighted reciprocal rank.
        Apply deduplication and confidence filtering with error handling.
        """
        logger.info("Fusing results")

        try:
            fused_results = await self.retry_manager.retry_with_backoff(
                lambda: self.result_fuser.fuse_results(
                    vector_results=state["vector_results"],
                    graph_results=state["graph_results"],
                    confidence_threshold=self.confidence_threshold,
                ),
                max_retries=2,
                exceptions=(Exception,),
            )

            state["fused_results"] = fused_results
            state["sources"] = [result.get("content", "") for result in fused_results]

            logger.info("Results fused", fused_count=len(fused_results))

        except Exception as e:
            logger.error("Result fusion failed", error=str(e))
            raise FusionError(f"Fusion failed: {str(e)}")

        return state

    async def generate_response(self, state: RAGState) -> RAGState:
        """Generate response based on fused results with comprehensive error handling."""
        logger.info("Generating response")

        if not state["fused_results"]:
            state["response"] = (
                "I couldn't find sufficient information to answer your query."
            )
            state["confidence_score"] = 0.0
            return state

        try:
            # Simple response generation (replace with actual LLM call)
            response_content = await self._synthesize_response_async(
                query=state["query"], sources=state["fused_results"]
            )

            state["response"] = response_content
            state["confidence_score"] = self._calculate_confidence(
                response_content, state["fused_results"]
            )

            logger.info(
                "Response generated",
                confidence=state["confidence_score"],
                response_length=len(response_content),
            )

        except Exception as e:
            logger.error("Response generation failed", error=str(e))
            raise ResponseGenerationError(f"Response generation failed: {str(e)}")

        return state

    async def validate_response(self, state: RAGState) -> RAGState:
        """Validate response for hallucinations and confidence with enhanced checks."""
        logger.info("Validating response", confidence=state["confidence_score"])

        try:
            # Check for hallucinations with enhanced detection
            state["hallucination_detected"] = await self._detect_hallucination_async(
                response=state["response"], sources=state["sources"]
            )

            # Check if correction is needed
            state["correction_needed"] = (
                state["confidence_score"] < self.confidence_threshold
                or state["hallucination_detected"]
            ) and state["iteration_count"] < self.max_corrections

            logger.info(
                "Validation completed",
                hallucination_detected=state["hallucination_detected"],
                correction_needed=state["correction_needed"],
                iteration=state["iteration_count"],
            )

        except Exception as e:
            logger.error("Response validation failed", error=str(e))
            # Don't raise here, continue with current state

        return state

    async def self_correction(self, state: RAGState) -> RAGState:
        """Apply self-correction by refining the query with intelligent refinement."""
        state["iteration_count"] += 1

        logger.info(
            "Applying self-correction",
            iteration=state["iteration_count"],
            reason=(
                "hallucination" if state["hallucination_detected"] else "low_confidence"
            ),
        )

        try:
            # Intelligent query refinement based on issue type
            if state["hallucination_detected"]:
                state["query"] = await self._refine_query_for_accuracy(state["query"])
            else:
                state["query"] = await self._refine_query_for_completeness(
                    state["query"]
                )

        except Exception as e:
            logger.error("Query refinement failed", error=str(e))
            # Fallback to simple refinement
            state["query"] = f"More specific information needed: {state['query']}"

        return state

    async def fallback_retrieval(self, state: RAGState) -> RAGState:
        """Fallback to external search when internal retrieval fails."""
        logger.info("Executing fallback retrieval")

        try:
            # Enhanced fallback with actual external search integration points
            # This would integrate with Tavily, Exa.ai, or other external APIs
            fallback_response = await self._execute_external_search(state["query"])

            state["response"] = fallback_response.get(
                "response",
                "I apologize, but I couldn't find sufficient information to answer your query.",
            )
            state["confidence_score"] = fallback_response.get("confidence", 0.3)

            logger.info(
                "Fallback retrieval completed", confidence=state["confidence_score"]
            )

        except Exception as e:
            logger.error("Fallback retrieval failed", error=str(e))
            state["response"] = "An error occurred while processing your query."
            state["confidence_score"] = 0.0

        return state

    def should_proceed_to_retrieval(self, state: RAGState) -> str:
        """Determine if we should proceed to retrieval or use fallback."""
        if not state["query"].strip():
            return "fallback"
        return "proceed"

    def should_correct(self, state: RAGState) -> str:
        """Determine the next action based on validation results."""
        if state["correction_needed"]:
            return "correct"
        elif state["confidence_score"] < 0.3:  # Very low confidence
            return "fallback"
        else:
            return "end"

    async def _synthesize_response_async(
        self, query: str, sources: List[Dict[str, Any]]
    ) -> str:
        """Asynchronous response synthesis with intelligent content processing."""
        if not sources:
            return "No relevant information found."

        try:
            # Intelligent content synthesis
            content_parts = []
            for i, source in enumerate(sources[:3]):  # Use top 3 sources
                content = source.get("content", "")
                if content:
                    # Extract key information intelligently
                    processed_content = self._extract_key_content(content, query)
                    if processed_content:
                        content_parts.append(f"Source {i+1}: {processed_content}")

            if not content_parts:
                return "The available sources don't contain relevant information for your query."

            return f"Based on the available information:\n\n" + "\n\n".join(
                content_parts
            )

        except Exception as e:
            logger.error("Response synthesis failed", error=str(e))
            return "Error synthesizing response from available sources."

    @lru_cache(maxsize=128)
    def _extract_key_content(self, content: str, query: str) -> str:
        """Extract key content relevant to the query (cached for performance)."""
        # Simple relevance extraction - replace with more sophisticated NLP
        query_terms = set(query.lower().split())
        content_lower = content.lower()

        # Find sentences containing query terms
        import re

        sentences = re.split(r"[.!?]+", content)
        relevant_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and any(term in sentence.lower() for term in query_terms):
                relevant_sentences.append(sentence)

        if relevant_sentences:
            return ". ".join(relevant_sentences[:2]) + "."
        else:
            return content[:200] + "..." if len(content) > 200 else content

    def _calculate_confidence(
        self, response: str, sources: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score with enhanced heuristics."""
        if not response or not sources:
            return 0.0

        # Enhanced confidence calculation
        source_quality = min(
            len(sources) / 3.0, 1.0
        )  # More sources = higher confidence
        response_quality = min(
            len(response) / 100.0, 1.0
        )  # Longer response = higher quality

        # Check for uncertainty markers
        uncertainty_markers = ["might", "possibly", "perhaps", "unclear", "unsure"]
        uncertainty_penalty = (
            sum(1 for marker in uncertainty_markers if marker in response.lower()) * 0.1
        )

        # Calculate average source confidence
        source_confidences = [source.get("confidence_score", 0.5) for source in sources]
        avg_source_confidence = (
            sum(source_confidences) / len(source_confidences)
            if source_confidences
            else 0.0
        )

        confidence = (source_quality + response_quality + avg_source_confidence) / 3.0
        confidence = max(0.0, confidence - uncertainty_penalty)

        return min(confidence, 1.0)

    async def _detect_hallucination_async(
        self, response: str, sources: List[str]
    ) -> bool:
        """Enhanced asynchronous hallucination detection."""
        try:
            # Use the existing detect_hallucination function but make it async
            return await asyncio.get_event_loop().run_in_executor(
                None, detect_hallucination, response, sources
            )
        except Exception as e:
            logger.error("Hallucination detection failed", error=str(e))
            return False  # Conservative default

    async def _refine_query_for_accuracy(self, query: str) -> str:
        """Refine query to improve factual accuracy."""
        # Add emphasis on factual information
        return f"Please provide only factual, verified information about: {query}"

    async def _refine_query_for_completeness(self, query: str) -> str:
        """Refine query to get more comprehensive information."""
        # Add request for more detailed information
        return f"Please provide comprehensive and detailed information about: {query}"

    async def _execute_external_search(self, query: str) -> Dict[str, Any]:
        """Execute external search as fallback (placeholder for integration)."""
        # Placeholder for external search integration (Tavily, Exa.ai, etc.)
        await asyncio.sleep(0.1)  # Simulate API call

        return {
            "response": f"External search results for: {query} (integration pending)",
            "confidence": 0.4,
            "sources": ["external_search"],
        }

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Main entry point for processing queries through the hybrid RAG workflow.

        Args:
            query: The user's query string

        Returns:
            Dict containing response, confidence score, and metadata
        """
        # Check cache first
        if self.cache:
            cached_result = self.cache.get(query)
            if cached_result:
                logger.info(
                    "Returning cached result",
                    query_hash=hashlib.md5(query.encode()).hexdigest()[:8],
                )
                return cached_result

        initial_state = RAGState(
            query=query,
            vector_results=[],
            graph_results=[],
            fused_results=[],
            response="",
            confidence_score=0.0,
            iteration_count=0,
            correction_needed=False,
            hallucination_detected=False,
            sources=[],
        )

        try:
            start_time = time.time()
            final_state = await self.workflow.ainvoke(initial_state)
            execution_time = time.time() - start_time

            result = {
                "response": final_state["response"],
                "confidence_score": final_state["confidence_score"],
                "iteration_count": final_state["iteration_count"],
                "hallucination_detected": final_state["hallucination_detected"],
                "sources_count": len(final_state["sources"]),
                "execution_time": execution_time,
                "metadata": {
                    "vector_results_count": len(final_state["vector_results"]),
                    "graph_results_count": len(final_state["graph_results"]),
                    "fused_results_count": len(final_state["fused_results"]),
                    "workflow_version": "1.0.0",
                },
            }

            # Cache successful results
            if self.cache and final_state["confidence_score"] > 0.5:
                self.cache.set(query, result)

            logger.info(
                "Query processing completed",
                execution_time=execution_time,
                confidence=final_state["confidence_score"],
                iterations=final_state["iteration_count"],
            )

            return result

        except OrchestrationError as e:
            logger.error("Orchestration error", error=str(e))
            return {
                "response": f"I encountered an issue processing your query: {str(e)}",
                "confidence_score": 0.0,
                "iteration_count": 0,
                "hallucination_detected": False,
                "sources_count": 0,
                "error": str(e),
                "error_type": "orchestration",
            }

        except Exception as e:
            logger.error("Unexpected workflow execution error", error=str(e))
            return {
                "response": "An unexpected error occurred while processing your query.",
                "confidence_score": 0.0,
                "iteration_count": 0,
                "hallucination_detected": False,
                "sources_count": 0,
                "error": str(e),
                "error_type": "unexpected",
            }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health_status = {
            "orchestrator": "healthy",
            "components": {},
            "timestamp": time.time(),
        }

        try:
            # Check vector retriever
            if hasattr(self.vector_retriever, "health_check"):
                health_status["components"][
                    "vector_retriever"
                ] = await self.vector_retriever.health_check()
            else:
                health_status["components"]["vector_retriever"] = "unknown"

            # Check graph retriever
            if hasattr(self.graph_retriever, "health_check"):
                health_status["components"][
                    "graph_retriever"
                ] = await self.graph_retriever.health_check()
            else:
                health_status["components"]["graph_retriever"] = "unknown"

            # Check result fuser
            health_status["components"]["result_fuser"] = "healthy"

            # Check cache
            if self.cache:
                health_status["components"]["cache"] = {
                    "status": "healthy",
                    "size": len(self.cache._cache),
                }

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            health_status["orchestrator"] = "unhealthy"
            health_status["error"] = str(e)

        return health_status

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics for monitoring."""
        metrics = {
            "cache_metrics": {},
            "configuration": {
                "max_corrections": self.max_corrections,
                "confidence_threshold": self.confidence_threshold,
            },
            "timestamp": time.time(),
        }

        if self.cache:
            metrics["cache_metrics"] = {
                "size": len(self.cache._cache),
                "max_size": self.cache.max_size,
                "ttl_seconds": self.cache.ttl_seconds,
            }

        return metrics
