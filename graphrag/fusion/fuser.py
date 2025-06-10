"""
Fusion Pipeline for Hybrid RAG System

This module implements the result fusion logic combining vector and graph retrieval results
using weighted reciprocal rank, content-based deduplication, and hallucination detection
according to the rhoSearcher ruleset.
"""

import hashlib
import re
from typing import List, Dict, Any, Set
from collections import defaultdict
import structlog

logger = structlog.get_logger(__name__)


def detect_hallucination(response: str, sources: List[str]) -> bool:
    """
    Detect potential hallucinations by comparing entities in response vs sources.

    Args:
        response: Generated response text
        sources: List of source content strings

    Returns:
        True if hallucination detected, False otherwise
    """
    try:
        if not response or not sources:
            return True  # No sources means potential hallucination

        # Extract entities from response and sources
        response_entities = extract_entities(response)
        source_entities = set()

        for source in sources:
            if source:  # Skip empty sources
                source_entities.update(extract_entities(source))

        # Check for novel entities in response not present in sources
        novel_entities = response_entities - source_entities

        # Calculate hallucination score
        if len(response_entities) == 0:
            return False  # No entities to check

        hallucination_ratio = len(novel_entities) / len(response_entities)

        # Consider it hallucination if >30% of entities are novel
        is_hallucination = hallucination_ratio > 0.3

        if is_hallucination:
            logger.warning(
                "Hallucination detected",
                novel_entities=list(novel_entities)[:5],  # Log first 5
                hallucination_ratio=hallucination_ratio,
                total_response_entities=len(response_entities),
            )

        return is_hallucination

    except Exception as e:
        logger.error("Hallucination detection failed", error=str(e))
        return True  # Err on the side of caution


def extract_entities(text: str) -> Set[str]:
    """
    Extract entities from text using simple heuristics.
    In production, this would use a proper NLP model.

    Args:
        text: Input text to extract entities from

    Returns:
        Set of extracted entity strings
    """
    try:
        if not text:
            return set()

        # Simple entity extraction using patterns
        entities = set()

        # Capitalized words (potential proper nouns)
        capitalized_words = re.findall(r"\b[A-Z][a-z]+\b", text)
        entities.update(capitalized_words)

        # Acronyms (2+ uppercase letters)
        acronyms = re.findall(r"\b[A-Z]{2,}\b", text)
        entities.update(acronyms)

        # Numbers with units (potential measurements, dates)
        numbers_with_units = re.findall(r"\b\d+(?:\.\d+)?\s*[a-zA-Z]+\b", text)
        entities.update(numbers_with_units)

        # Technical terms (words with specific patterns)
        technical_terms = re.findall(r"\b[a-z]+(?:-[a-z]+)+\b", text)
        entities.update(technical_terms)

        # Filter out common stop words
        stop_words = {
            "The",
            "This",
            "That",
            "These",
            "Those",
            "A",
            "An",
            "And",
            "Or",
            "But",
            "In",
            "On",
            "At",
            "To",
            "For",
            "Of",
            "With",
            "By",
            "From",
            "As",
            "Is",
            "Are",
            "Was",
            "Were",
            "Be",
            "Been",
            "Have",
            "Has",
            "Had",
            "Do",
            "Does",
            "Did",
            "Will",
            "Would",
            "Could",
            "Should",
            "May",
            "Might",
            "Must",
            "Can",
        }

        entities = {
            entity
            for entity in entities
            if entity not in stop_words and len(entity) > 2
        }

        return entities

    except Exception as e:
        logger.error("Entity extraction failed", error=str(e))
        return set()


class ResultFuser:
    """
    Production-ready result fusion implementing weighted reciprocal rank (RRF)
    with content deduplication and confidence filtering.
    """

    def __init__(self):
        # Fusion parameters according to ruleset
        self.vector_weight = 0.6  # Slightly favor vector results
        self.graph_weight = 0.4  # Graph results for context
        self.rrf_k = 60  # Reciprocal rank fusion parameter
        self.similarity_threshold = (
            0.8  # Content similarity threshold for deduplication
        )

    async def fuse_results(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        confidence_threshold: float = 0.65,
    ) -> List[Dict[str, Any]]:
        """
        Fuse vector and graph results using weighted reciprocal rank.

        Args:
            vector_results: Results from vector retrieval
            graph_results: Results from graph traversal
            confidence_threshold: Minimum confidence for result inclusion

        Returns:
            List of fused and ranked results
        """
        try:
            logger.info(
                "Starting result fusion",
                vector_count=len(vector_results),
                graph_count=len(graph_results),
                confidence_threshold=confidence_threshold,
            )

            # Normalize scores for both result types
            normalized_vector = self._normalize_scores(vector_results, "vector")
            normalized_graph = self._normalize_scores(graph_results, "graph")

            # Apply weighted reciprocal rank fusion
            fused_scores = self._apply_weighted_rrf(normalized_vector, normalized_graph)

            # Convert to result objects with metadata
            fused_results = self._create_fused_results(
                fused_scores, normalized_vector, normalized_graph
            )

            # Apply content-based deduplication
            deduplicated_results = self._deduplicate_results(fused_results)

            # Filter by confidence threshold
            filtered_results = [
                result
                for result in deduplicated_results
                if result.get("confidence_score", 0.0) >= confidence_threshold
            ]

            # Sort by final fusion score
            filtered_results.sort(
                key=lambda x: x.get("fusion_score", 0.0), reverse=True
            )

            logger.info(
                "Result fusion completed",
                fused_count=len(fused_results),
                deduplicated_count=len(deduplicated_results),
                final_count=len(filtered_results),
            )

            return filtered_results

        except Exception as e:
            logger.error("Result fusion failed", error=str(e))
            return []

    def _normalize_scores(
        self, results: List[Dict[str, Any]], result_type: str
    ) -> List[Dict[str, Any]]:
        """Normalize scores to 0-1 range and add metadata."""
        if not results:
            return []

        try:
            # Extract scores based on result type
            if result_type == "vector":
                scores = [result.get("score", 0.0) for result in results]
            else:  # graph
                scores = [result.get("relevance_score", 0.0) for result in results]

            # Normalize scores to 0-1 range
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                score_range = max_score - min_score

                if score_range > 0:
                    normalized_scores = [
                        (score - min_score) / score_range for score in scores
                    ]
                else:
                    normalized_scores = [1.0] * len(scores)
            else:
                normalized_scores = []

            # Create normalized results
            normalized_results = []
            for i, result in enumerate(results):
                normalized_result = result.copy()
                normalized_result["normalized_score"] = (
                    normalized_scores[i] if i < len(normalized_scores) else 0.0
                )
                normalized_result["result_type"] = result_type
                normalized_result["original_rank"] = i + 1
                normalized_results.append(normalized_result)

            return normalized_results

        except Exception as e:
            logger.error(
                "Score normalization failed", error=str(e), result_type=result_type
            )
            return results

    def _apply_weighted_rrf(
        self, vector_results: List[Dict[str, Any]], graph_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Apply weighted reciprocal rank fusion."""
        try:
            fused_scores = defaultdict(float)

            # Process vector results
            for rank, result in enumerate(vector_results, 1):
                content_hash = self._get_content_hash(result)
                rrf_score = self.vector_weight / (self.rrf_k + rank)
                fused_scores[content_hash] += rrf_score

            # Process graph results
            for rank, result in enumerate(graph_results, 1):
                content_hash = self._get_content_hash(result)
                rrf_score = self.graph_weight / (self.rrf_k + rank)
                fused_scores[content_hash] += rrf_score

            return dict(fused_scores)

        except Exception as e:
            logger.error("Weighted RRF application failed", error=str(e))
            return {}

    def _get_content_hash(self, result: Dict[str, Any]) -> str:
        """Generate content hash for deduplication."""
        try:
            # Get content from different possible fields
            content = (
                result.get("content", "")
                or result.get("entity", "")
                or " ".join(result.get("entity_names", []))
                or str(result.get("id", ""))
            )

            # Normalize content for hashing
            normalized_content = re.sub(r"\s+", " ", content.lower().strip())

            # Generate hash
            return hashlib.md5(normalized_content.encode()).hexdigest()

        except Exception as e:
            logger.error("Content hash generation failed", error=str(e))
            return str(hash(str(result)))

    def _create_fused_results(
        self,
        fused_scores: Dict[str, float],
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Create final fused result objects."""
        try:
            # Create lookup for results by content hash
            result_lookup = {}

            for result in vector_results + graph_results:
                content_hash = self._get_content_hash(result)
                if content_hash not in result_lookup:
                    result_lookup[content_hash] = result
                else:
                    # Merge metadata from multiple sources
                    existing = result_lookup[content_hash]
                    existing["source_types"] = existing.get("source_types", []) + [
                        result.get("result_type", "unknown")
                    ]

            # Create fused results
            fused_results = []
            for content_hash, fusion_score in fused_scores.items():
                if content_hash in result_lookup:
                    result = result_lookup[content_hash].copy()
                    result["fusion_score"] = fusion_score
                    result["confidence_score"] = min(
                        result.get("normalized_score", 0.0) + (fusion_score * 0.5), 1.0
                    )
                    fused_results.append(result)

            return fused_results

        except Exception as e:
            logger.error("Fused result creation failed", error=str(e))
            return []

    def _deduplicate_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate results based on content similarity."""
        try:
            if not results:
                return []

            deduplicated = []
            seen_hashes = set()

            for result in results:
                content_hash = self._get_content_hash(result)

                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    deduplicated.append(result)
                else:
                    # Log deduplication
                    logger.debug(
                        "Duplicate result removed", content_hash=content_hash[:8]
                    )

            logger.info(
                "Deduplication completed",
                original_count=len(results),
                deduplicated_count=len(deduplicated),
                removed_count=len(results) - len(deduplicated),
            )

            return deduplicated

        except Exception as e:
            logger.error("Deduplication failed", error=str(e))
            return results

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity using simple string metrics."""
        try:
            if not content1 or not content2:
                return 0.0

            # Normalize content
            c1 = set(content1.lower().split())
            c2 = set(content2.lower().split())

            # Jaccard similarity
            intersection = len(c1.intersection(c2))
            union = len(c1.union(c2))

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            logger.error("Content similarity calculation failed", error=str(e))
            return 0.0

    def get_fusion_statistics(
        self, fused_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get statistics about the fusion process."""
        try:
            if not fused_results:
                return {}

            # Count by source type
            source_type_counts = defaultdict(int)
            confidence_scores = []
            fusion_scores = []

            for result in fused_results:
                source_types = result.get(
                    "source_types", [result.get("result_type", "unknown")]
                )
                for source_type in source_types:
                    source_type_counts[source_type] += 1

                confidence_scores.append(result.get("confidence_score", 0.0))
                fusion_scores.append(result.get("fusion_score", 0.0))

            return {
                "total_results": len(fused_results),
                "source_type_distribution": dict(source_type_counts),
                "avg_confidence_score": sum(confidence_scores) / len(confidence_scores),
                "avg_fusion_score": sum(fusion_scores) / len(fusion_scores),
                "max_confidence_score": max(confidence_scores),
                "max_fusion_score": max(fusion_scores),
                "min_confidence_score": min(confidence_scores),
                "min_fusion_score": min(fusion_scores),
            }

        except Exception as e:
            logger.error("Failed to calculate fusion statistics", error=str(e))
            return {}
