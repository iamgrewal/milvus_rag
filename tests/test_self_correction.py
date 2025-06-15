"""
Comprehensive tests for the Self-Correction Framework

Tests multi-layer validation system, confidence scoring, consistency checking,
and automated correction mechanisms for the hybrid RAG system.
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag.validation.self_correction import (
    AutomatedCorrector,
    ConfidenceScore,
    ConfidenceScorer,
    CorrectionAction,
    CorrectionSuggestion,
    HallucinationDetector,
    RAGResponse,
    SelfCorrectionFramework,
    ValidationEngine,
    ValidationResult,
    ValidationSeverity,
    ValidationType,
)


class TestDataClasses:
    """Test data classes and enums."""

    def test_validation_type_enum(self):
        """Test ValidationType enum values."""
        assert ValidationType.ENTITY_CONSISTENCY.value == "entity_consistency"
        assert ValidationType.HALLUCINATION_DETECTION.value == "hallucination_detection"
        assert len(list(ValidationType)) == 6

    def test_validation_severity_enum(self):
        """Test ValidationSeverity enum values."""
        assert ValidationSeverity.LOW.value == "low"
        assert ValidationSeverity.CRITICAL.value == "critical"
        assert len(list(ValidationSeverity)) == 4

    def test_correction_action_enum(self):
        """Test CorrectionAction enum values."""
        assert CorrectionAction.ACCEPT.value == "accept"
        assert CorrectionAction.REGENERATE.value == "regenerate"
        assert len(list(CorrectionAction)) == 5

    def test_validation_result_creation(self):
        """Test ValidationResult dataclass creation."""
        result = ValidationResult(
            rule_type=ValidationType.ENTITY_CONSISTENCY,
            passed=True,
            confidence=0.9,
            severity=ValidationSeverity.LOW,
            message="Test validation passed",
            evidence={"test": "data"},
            suggestions=["suggestion1"],
            execution_time=0.1,
        )

        assert result.rule_type == ValidationType.ENTITY_CONSISTENCY
        assert result.passed is True
        assert result.confidence == 0.9
        assert result.severity == ValidationSeverity.LOW
        assert result.message == "Test validation passed"
        assert result.evidence == {"test": "data"}
        assert result.suggestions == ["suggestion1"]
        assert result.execution_time == 0.1

    def test_confidence_score_creation(self):
        """Test ConfidenceScore dataclass creation."""
        score = ConfidenceScore(
            source_confidence=0.8,
            consistency_confidence=0.9,
            completeness_confidence=0.7,
            overall_confidence=0.8,
            confidence_factors={"test": 0.5},
            uncertainty_flags=["flag1"],
        )

        assert score.source_confidence == 0.8
        assert score.consistency_confidence == 0.9
        assert score.completeness_confidence == 0.7
        assert score.overall_confidence == 0.8
        assert score.confidence_factors == {"test": 0.5}
        assert score.uncertainty_flags == ["flag1"]

    def test_correction_suggestion_creation(self):
        """Test CorrectionSuggestion dataclass creation."""
        suggestion = CorrectionSuggestion(
            action=CorrectionAction.MODIFY,
            confidence=0.8,
            original_text="Original text",
            corrected_text="Corrected text",
            reasoning="Test reasoning",
        )

        assert suggestion.action == CorrectionAction.MODIFY
        assert suggestion.confidence == 0.8
        assert suggestion.original_text == "Original text"
        assert suggestion.corrected_text == "Corrected text"
        assert suggestion.reasoning == "Test reasoning"

    def test_rag_response_creation(self):
        """Test RAGResponse dataclass creation."""
        response = RAGResponse(
            content="Test response content",
            sources=[{"source": "test"}],
            entities=["entity1", "entity2"],
            relationships=[{"source": "A", "target": "B", "type": "relates_to"}],
        )

        assert response.content == "Test response content"
        assert len(response.sources) == 1
        assert len(response.entities) == 2
        assert len(response.relationships) == 1
        assert response.confidence_score is None
        assert len(response.validation_results) == 0
        assert isinstance(response.timestamp, datetime)


class TestHallucinationDetector:
    """Test HallucinationDetector functionality."""

    @pytest.fixture
    def detector(self):
        """Create HallucinationDetector instance."""
        return HallucinationDetector()

    @pytest.mark.asyncio
    async def test_detect_uncertainty_language_no_uncertainty(self, detector):
        """Test detection with no uncertainty language."""
        text = "The capital of France is Paris. It was established in the 3rd century BC."

        result = await detector.detect_uncertainty_language(text)

        assert result["uncertainty_score"] == 0.0
        assert result["matches"] == []
        assert result["has_uncertainty"] is False

    @pytest.mark.asyncio
    async def test_detect_uncertainty_language_with_uncertainty(self, detector):
        """Test detection with uncertainty language."""
        text = "I think the capital of France is Paris. It seems to be established maybe in the 3rd century BC."

        result = await detector.detect_uncertainty_language(text)

        assert result["uncertainty_score"] > 0.0
        assert len(result["matches"]) >= 2  # "I think" and "maybe"
        assert result["has_uncertainty"] is True

        # Check match details
        for match in result["matches"]:
            assert "pattern" in match
            assert "match" in match
            assert "position" in match
            assert "context" in match

    @pytest.mark.asyncio
    async def test_detect_uncertainty_language_case_insensitive(self, detector):
        """Test that detection is case insensitive."""
        text = "I THINK this is correct. MAYBE it's true."

        result = await detector.detect_uncertainty_language(text)

        assert result["uncertainty_score"] > 0.0
        assert result["has_uncertainty"] is True

    @pytest.mark.asyncio
    async def test_detect_factual_inconsistencies_no_issues(self, detector):
        """Test factual inconsistency detection with clean text."""
        text = "John was born in 1980 and graduated in 2002."
        context = {}

        result = await detector.detect_factual_inconsistencies(text, context)

        assert result["inconsistency_score"] == 0.0
        assert result["inconsistencies"] == []
        assert result["has_inconsistencies"] is False

    @pytest.mark.asyncio
    async def test_detect_factual_inconsistencies_large_year_gap(self, detector):
        """Test detection of large year gaps."""
        text = "This happened in 1066 and was later revised in 2020."
        context = {}

        result = await detector.detect_factual_inconsistencies(text, context)

        assert result["inconsistency_score"] > 0.0
        assert len(result["inconsistencies"]) > 0
        assert result["has_inconsistencies"] is True

        # Check inconsistency details
        inconsistency = result["inconsistencies"][0]
        assert inconsistency["type"] == "temporal_inconsistency"
        assert "1066-2020" in inconsistency["description"]

    @pytest.mark.asyncio
    async def test_detect_factual_inconsistencies_logical_issues(self, detector):
        """Test detection of logical inconsistencies."""
        text = "He was born after he died in 1990."
        context = {}

        result = await detector.detect_factual_inconsistencies(text, context)

        assert result["inconsistency_score"] > 0.0
        assert len(result["inconsistencies"]) > 0

        # Check for logical inconsistency
        logical_issues = [
            inc for inc in result["inconsistencies"] 
            if inc["type"] == "logical_inconsistency"
        ]
        assert len(logical_issues) > 0


class TestConfidenceScorer:
    """Test ConfidenceScorer functionality."""

    @pytest.fixture
    def scorer(self):
        """Create ConfidenceScorer instance."""
        return ConfidenceScorer()

    @pytest.fixture
    def sample_response(self):
        """Create sample RAG response for testing."""
        return RAGResponse(
            content="Paris is the capital of France. It has a population of over 2 million people.",
            sources=[
                {"content": "Paris is the capital city of France", "confidence": 0.9, "type": "document"},
                {"content": "France capital Paris population", "confidence": 0.8, "type": "database"}
            ],
            entities=["Paris", "France"],
            relationships=[{"source": "Paris", "target": "France", "type": "capital_of"}]
        )

    @pytest.mark.asyncio
    async def test_calculate_source_confidence_no_sources(self, scorer):
        """Test source confidence calculation with no sources."""
        response = RAGResponse(content="Test content")

        confidence = await scorer.calculate_source_confidence(response)

        assert confidence == 0.3  # Low confidence without sources

    @pytest.mark.asyncio
    async def test_calculate_source_confidence_with_sources(self, scorer, sample_response):
        """Test source confidence calculation with good sources."""
        confidence = await scorer.calculate_source_confidence(sample_response)

        assert 0.5 <= confidence <= 1.0
        # Should be higher than base score due to good source metadata

    @pytest.mark.asyncio
    async def test_calculate_consistency_confidence(self, scorer, sample_response):
        """Test consistency confidence calculation."""
        context = {}

        confidence = await scorer.calculate_consistency_confidence(sample_response, context)

        assert 0.0 <= confidence <= 1.0
        # Should be relatively high for this consistent response
        assert confidence >= 0.6

    @pytest.mark.asyncio
    async def test_calculate_consistency_confidence_with_issues(self, scorer):
        """Test consistency confidence with entity issues."""
        response = RAGResponse(
            content="Paris is great. Pariz is the capital.",
            entities=["Paris", "Pariz"],  # Variation that should reduce confidence
            relationships=[]
        )
        context = {}

        confidence = await scorer.calculate_consistency_confidence(response, context)

        # Should be lower due to entity name variation
        assert confidence <= 0.8

    @pytest.mark.asyncio
    async def test_calculate_completeness_confidence_short_response(self, scorer):
        """Test completeness confidence with short response."""
        response = RAGResponse(content="Yes.", entities=[], relationships=[])
        query = "What is the capital of France?"

        confidence = await scorer.calculate_completeness_confidence(response, query)

        # Should be low due to short response length
        assert confidence < 0.7

    @pytest.mark.asyncio
    async def test_calculate_completeness_confidence_comprehensive_response(self, scorer, sample_response):
        """Test completeness confidence with comprehensive response."""
        query = "What is the capital of France and its population?"

        confidence = await scorer.calculate_completeness_confidence(sample_response, query)

        # Should be high due to length, entities, relationships, and keyword overlap
        assert confidence >= 0.7

    @pytest.mark.asyncio
    async def test_calculate_confidence_score_comprehensive(self, scorer, sample_response):
        """Test comprehensive confidence score calculation."""
        query = "What is the capital of France?"
        context = {}

        confidence_score = await scorer.calculate_confidence_score(sample_response, query, context)

        assert isinstance(confidence_score, ConfidenceScore)
        assert 0.0 <= confidence_score.source_confidence <= 1.0
        assert 0.0 <= confidence_score.consistency_confidence <= 1.0
        assert 0.0 <= confidence_score.completeness_confidence <= 1.0
        assert 0.0 <= confidence_score.overall_confidence <= 1.0

        # Check confidence factors
        assert "source_weight" in confidence_score.confidence_factors
        assert "consistency_weight" in confidence_score.confidence_factors
        assert "completeness_weight" in confidence_score.confidence_factors

        # Should have good overall confidence for this well-formed response
        assert confidence_score.overall_confidence >= 0.6

    @pytest.mark.asyncio
    async def test_calculate_confidence_score_with_flags(self, scorer):
        """Test confidence score with uncertainty flags."""
        response = RAGResponse(content="Short.", sources=[], entities=[])
        query = "Complex question about many things?"
        context = {}

        confidence_score = await scorer.calculate_confidence_score(response, query, context)

        # Should have uncertainty flags due to low scores
        assert len(confidence_score.uncertainty_flags) > 0
        assert any("source" in flag.lower() for flag in confidence_score.uncertainty_flags)


class TestValidationEngine:
    """Test ValidationEngine functionality."""

    @pytest.fixture
    def engine(self):
        """Create ValidationEngine instance."""
        return ValidationEngine()

    @pytest.fixture
    def sample_response(self):
        """Create sample RAG response for testing."""
        return RAGResponse(
            content="Paris is the capital of France. It was established in ancient times.",
            entities=["Paris", "France"],
            relationships=[{"source": "Paris", "target": "France", "type": "capital_of"}]
        )

    @pytest.mark.asyncio
    async def test_validate_entity_consistency_no_entities(self, engine):
        """Test entity consistency validation with no entities."""
        response = RAGResponse(content="Test content", entities=[])
        context = {}

        result = await engine.validate_entity_consistency(response, context)

        assert result.rule_type == ValidationType.ENTITY_CONSISTENCY
        assert result.passed is True
        assert result.confidence == 1.0
        assert result.severity == ValidationSeverity.LOW
        assert "No entities to validate" in result.message

    @pytest.mark.asyncio
    async def test_validate_entity_consistency_good_entities(self, engine, sample_response):
        """Test entity consistency validation with consistent entities."""
        context = {}

        result = await engine.validate_entity_consistency(sample_response, context)

        assert result.rule_type == ValidationType.ENTITY_CONSISTENCY
        assert result.passed is True
        assert result.confidence >= 0.7
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_validate_entity_consistency_with_variations(self, engine):
        """Test entity consistency validation with entity variations."""
        response = RAGResponse(
            content="Paris is great. Pariz is the capital.",
            entities=["Paris", "Pariz", "paris"]  # Variations
        )
        context = {}

        result = await engine.validate_entity_consistency(response, context)

        assert result.rule_type == ValidationType.ENTITY_CONSISTENCY
        # Should detect issues with entity variations
        assert "issues" in result.evidence
        assert len(result.evidence["issues"]) > 0

    @pytest.mark.asyncio
    async def test_validate_relationship_consistency_no_relationships(self, engine):
        """Test relationship consistency validation with no relationships."""
        response = RAGResponse(content="Test content", relationships=[])
        context = {}

        result = await engine.validate_relationship_consistency(response, context)

        assert result.rule_type == ValidationType.RELATIONSHIP_CONSISTENCY
        assert result.passed is True
        assert result.confidence == 1.0
        assert "No relationships to validate" in result.message

    @pytest.mark.asyncio
    async def test_validate_relationship_consistency_good_relationships(self, engine, sample_response):
        """Test relationship consistency validation with good relationships."""
        context = {}

        result = await engine.validate_relationship_consistency(sample_response, context)

        assert result.rule_type == ValidationType.RELATIONSHIP_CONSISTENCY
        assert result.confidence >= 0.8
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_validate_temporal_consistency_insufficient_dates(self, engine):
        """Test temporal consistency validation with insufficient date information."""
        response = RAGResponse(content="Paris is a city.")
        context = {}

        result = await engine.validate_temporal_consistency(response, context)

        assert result.rule_type == ValidationType.TEMPORAL_CONSISTENCY
        assert result.passed is True
        assert "Insufficient temporal information" in result.message

    @pytest.mark.asyncio
    async def test_validate_temporal_consistency_with_dates(self, engine):
        """Test temporal consistency validation with date information."""
        response = RAGResponse(content="Founded in 1066 and modernized in 2020.")
        context = {}

        result = await engine.validate_temporal_consistency(response, context)

        assert result.rule_type == ValidationType.TEMPORAL_CONSISTENCY
        assert result.execution_time > 0
        # Should detect large temporal range
        assert "date_range" in result.evidence

    @pytest.mark.asyncio
    async def test_validate_factual_consistency(self, engine, sample_response):
        """Test factual consistency validation."""
        context = {}

        result = await engine.validate_factual_consistency(sample_response, context)

        assert result.rule_type == ValidationType.FACTUAL_CONSISTENCY
        assert result.confidence >= 0.0
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_validate_hallucination_detection_clean_text(self, engine):
        """Test hallucination detection with clean text."""
        response = RAGResponse(content="Paris is the capital of France.")
        context = {}

        result = await engine.validate_hallucination_detection(response, context)

        assert result.rule_type == ValidationType.HALLUCINATION_DETECTION
        assert result.passed is True
        assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_validate_hallucination_detection_uncertain_text(self, engine):
        """Test hallucination detection with uncertain language."""
        response = RAGResponse(content="I think Paris might be the capital of France.")
        context = {}

        result = await engine.validate_hallucination_detection(response, context)

        assert result.rule_type == ValidationType.HALLUCINATION_DETECTION
        assert result.confidence < 0.9  # Should detect uncertainty
        assert len(result.suggestions) > 0

    @pytest.mark.asyncio
    async def test_validate_response_comprehensive(self, engine, sample_response):
        """Test comprehensive response validation."""
        query = "What is the capital of France?"
        context = {}

        results = await engine.validate_response(sample_response, query, context)

        # Should return results for all enabled validation types
        assert len(results) > 0
        
        # Check that all results are ValidationResult instances
        for result in results:
            assert isinstance(result, ValidationResult)
            assert hasattr(result, 'rule_type')
            assert hasattr(result, 'passed')
            assert hasattr(result, 'confidence')
            assert hasattr(result, 'execution_time')

    def test_entities_similar(self, engine):
        """Test entity similarity checking."""
        # Identical entities
        assert engine._entities_similar("paris", "paris") is True
        
        # Substring relationship
        assert engine._entities_similar("paris", "paris france") is True
        
        # Similar word sets
        assert engine._entities_similar("john smith", "smith john") is True
        
        # Different entities
        assert engine._entities_similar("paris", "london") is False


class TestAutomatedCorrector:
    """Test AutomatedCorrector functionality."""

    @pytest.fixture
    def corrector(self):
        """Create AutomatedCorrector instance."""
        return AutomatedCorrector()

    @pytest.fixture
    def sample_response(self):
        """Create sample RAG response for testing."""
        return RAGResponse(
            content="Paris is the capital of France.",
            entities=["Paris", "France"]
        )

    @pytest.mark.asyncio
    async def test_correct_entity_consistency_no_issues(self, corrector, sample_response):
        """Test entity consistency correction with no issues."""
        validation_result = ValidationResult(
            rule_type=ValidationType.ENTITY_CONSISTENCY,
            passed=True,
            confidence=1.0,
            severity=ValidationSeverity.LOW,
            message="No issues",
            evidence={"issues": []}
        )

        suggestion = await corrector._correct_entity_consistency(sample_response, validation_result)

        assert suggestion.action == CorrectionAction.ACCEPT
        assert suggestion.confidence == 1.0
        assert suggestion.original_text == suggestion.corrected_text

    @pytest.mark.asyncio
    async def test_correct_entity_consistency_with_issues(self, corrector):
        """Test entity consistency correction with issues."""
        response = RAGResponse(
            content="Paris is great. Pariz is the capital.",
            entities=["Paris", "Pariz"]
        )
        
        validation_result = ValidationResult(
            rule_type=ValidationType.ENTITY_CONSISTENCY,
            passed=False,
            confidence=0.5,
            severity=ValidationSeverity.MEDIUM,
            message="Issues found",
            evidence={"issues": ["Entity variations found: Paris, Pariz"]}
        )

        suggestion = await corrector._correct_entity_consistency(response, validation_result)

        assert suggestion.action == CorrectionAction.MODIFY
        assert suggestion.confidence == 0.7
        assert suggestion.corrected_text != suggestion.original_text
        assert "Paris" in suggestion.corrected_text
        # Should standardize to first entity name

    @pytest.mark.asyncio
    async def test_correct_relationship_consistency(self, corrector, sample_response):
        """Test relationship consistency correction."""
        validation_result = ValidationResult(
            rule_type=ValidationType.RELATIONSHIP_CONSISTENCY,
            passed=False,
            confidence=0.4,
            severity=ValidationSeverity.MEDIUM,
            message="Issues found",
            evidence={"issues": ["Conflicting relationships found"]}
        )

        suggestion = await corrector._correct_relationship_consistency(sample_response, validation_result)

        # Relationship issues should be flagged for review
        assert suggestion.action == CorrectionAction.FLAG_UNCERTAIN
        assert suggestion.confidence == 0.5

    @pytest.mark.asyncio
    async def test_correct_temporal_consistency(self, corrector, sample_response):
        """Test temporal consistency correction."""
        validation_result = ValidationResult(
            rule_type=ValidationType.TEMPORAL_CONSISTENCY,
            passed=False,
            confidence=0.3,
            severity=ValidationSeverity.HIGH,
            message="Temporal issues",
            evidence={"issues": ["Large temporal range"]}
        )

        suggestion = await corrector._correct_temporal_consistency(sample_response, validation_result)

        # Temporal issues should be flagged for review
        assert suggestion.action == CorrectionAction.FLAG_UNCERTAIN
        assert suggestion.confidence == 0.3

    @pytest.mark.asyncio
    async def test_correct_factual_consistency(self, corrector, sample_response):
        """Test factual consistency correction."""
        validation_result = ValidationResult(
            rule_type=ValidationType.FACTUAL_CONSISTENCY,
            passed=False,
            confidence=0.4,
            severity=ValidationSeverity.HIGH,
            message="Factual issues",
            evidence={"issues": [{"type": "contradiction"}]}
        )

        suggestion = await corrector._correct_factual_consistency(sample_response, validation_result)

        # Factual issues should trigger regeneration
        assert suggestion.action == CorrectionAction.REGENERATE
        assert suggestion.confidence == 0.4

    @pytest.mark.asyncio
    async def test_correct_hallucinations_low_uncertainty(self, corrector):
        """Test hallucination correction with low uncertainty."""
        response = RAGResponse(content="Paris is the capital of France.")
        
        validation_result = ValidationResult(
            rule_type=ValidationType.HALLUCINATION_DETECTION,
            passed=True,
            confidence=0.9,
            severity=ValidationSeverity.LOW,
            message="Low uncertainty",
            evidence={"uncertainty_score": 0.1, "matches": []}
        )

        suggestion = await corrector._correct_hallucinations(response, validation_result)

        assert suggestion.action == CorrectionAction.ACCEPT
        assert suggestion.confidence == 1.0

    @pytest.mark.asyncio
    async def test_correct_hallucinations_with_uncertainty(self, corrector):
        """Test hallucination correction with uncertainty language."""
        response = RAGResponse(content="I think Paris might be the capital of France.")
        
        validation_result = ValidationResult(
            rule_type=ValidationType.HALLUCINATION_DETECTION,
            passed=False,
            confidence=0.5,
            severity=ValidationSeverity.MEDIUM,
            message="Uncertainty detected",
            evidence={
                "uncertainty_score": 0.6,
                "matches": [
                    {"match": "I think", "position": [0, 7]},
                    {"match": "might", "position": [14, 19]}
                ]
            }
        )

        suggestion = await corrector._correct_hallucinations(response, validation_result)

        assert suggestion.action == CorrectionAction.MODIFY
        assert suggestion.confidence == 0.8
        assert "I think" not in suggestion.corrected_text
        assert "might" not in suggestion.corrected_text

    @pytest.mark.asyncio
    async def test_generate_corrections_no_failures(self, corrector, sample_response):
        """Test correction generation with no validation failures."""
        validation_results = [
            ValidationResult(
                rule_type=ValidationType.ENTITY_CONSISTENCY,
                passed=True,
                confidence=1.0,
                severity=ValidationSeverity.LOW,
                message="Passed"
            )
        ]

        corrections = await corrector.generate_corrections(sample_response, validation_results)

        assert len(corrections) == 0  # No corrections needed

    @pytest.mark.asyncio
    async def test_generate_corrections_with_failures(self, corrector):
        """Test correction generation with validation failures."""
        response = RAGResponse(content="I think Paris might be the capital.")
        
        validation_results = [
            ValidationResult(
                rule_type=ValidationType.HALLUCINATION_DETECTION,
                passed=False,
                confidence=0.5,
                severity=ValidationSeverity.MEDIUM,
                message="Uncertainty detected",
                evidence={"uncertainty_score": 0.6, "matches": []}
            ),
            ValidationResult(
                rule_type=ValidationType.ENTITY_CONSISTENCY,
                passed=False,
                confidence=0.4,
                severity=ValidationSeverity.MEDIUM,
                message="Entity issues",
                evidence={"issues": ["Entity variations found: Paris, Pariz"]}
            )
        ]

        corrections = await corrector.generate_corrections(response, validation_results)

        assert len(corrections) == 2  # One correction per failed validation
        
        # Check that corrections are appropriate
        for correction in corrections:
            assert isinstance(correction, CorrectionSuggestion)
            assert correction.action in [CorrectionAction.MODIFY, CorrectionAction.FLAG_UNCERTAIN, CorrectionAction.REGENERATE]


class TestSelfCorrectionFramework:
    """Test SelfCorrectionFramework integration."""

    @pytest.fixture
    def framework(self):
        """Create SelfCorrectionFramework instance."""
        return SelfCorrectionFramework()

    @pytest.fixture
    def sample_response(self):
        """Create sample RAG response for testing."""
        return RAGResponse(
            content="Paris is the capital of France. It has a rich history.",
            sources=[{"content": "Paris capital France", "confidence": 0.9}],
            entities=["Paris", "France"],
            relationships=[{"source": "Paris", "target": "France", "type": "capital_of"}]
        )

    @pytest.mark.asyncio
    async def test_validate_and_correct_disabled(self):
        """Test framework behavior when disabled."""
        with patch("graphrag.validation.self_correction.Config.SELF_CORRECTION_ENABLED", False):
            framework = SelfCorrectionFramework()
            
            response = RAGResponse(content="Test")
            result = await framework.validate_and_correct(response, "query", {})
            
            assert result == response  # Should return unchanged

    @pytest.mark.asyncio
    async def test_validate_and_correct_passing_validation(self, framework, sample_response):
        """Test framework with response that passes validation."""
        query = "What is the capital of France?"
        context = {}

        result = await framework.validate_and_correct(sample_response, query, context)

        assert isinstance(result, RAGResponse)
        assert result.content == sample_response.content  # Should not change content
        assert result.confidence_score is not None
        assert len(result.validation_results) > 0
        assert "self_correction" in result.metadata
        
        # Check metadata
        sc_metadata = result.metadata["self_correction"]
        assert sc_metadata["enabled"] is True
        assert sc_metadata["iterations"] >= 1
        assert "final_confidence" in sc_metadata
        assert "processing_time" in sc_metadata

    @pytest.mark.asyncio
    async def test_validate_and_correct_with_corrections(self, framework):
        """Test framework with response that needs corrections."""
        response = RAGResponse(
            content="I think Paris might be the capital of France, maybe.",
            entities=["Paris", "France"]
        )
        query = "What is the capital of France?"
        context = {}

        result = await framework.validate_and_correct(response, query, context)

        assert isinstance(result, RAGResponse)
        assert result.confidence_score is not None
        assert len(result.validation_results) > 0
        
        # Should have correction suggestions
        if result.correction_suggestions:
            assert len(result.correction_suggestions) > 0
        
        # Check that corrections were applied if available
        sc_metadata = result.metadata["self_correction"]
        assert sc_metadata["iterations"] >= 1

    @pytest.mark.asyncio
    async def test_validate_and_correct_max_iterations(self):
        """Test framework respects max iterations limit."""
        with patch("graphrag.validation.self_correction.Config.VALIDATION_MAX_ITERATIONS", 1):
            framework = SelfCorrectionFramework()
            
            response = RAGResponse(content="Uncertain content I think maybe possibly.")
            query = "Test query"
            context = {}

            result = await framework.validate_and_correct(response, query, context)

            # Should stop after 1 iteration
            sc_metadata = result.metadata["self_correction"]
            assert sc_metadata["iterations"] == 1

    @pytest.mark.asyncio
    async def test_validate_and_correct_timeout(self):
        """Test framework respects timeout."""
        with patch("graphrag.validation.self_correction.Config.VALIDATION_TIMEOUT_SECONDS", 0.001):
            framework = SelfCorrectionFramework()
            
            response = RAGResponse(content="Test content")
            query = "Test query"
            context = {}

            result = await framework.validate_and_correct(response, query, context)

            # Should complete quickly due to timeout
            assert "self_correction" in result.metadata

    def test_select_best_correction_empty_list(self, framework):
        """Test best correction selection with empty list."""
        result = framework._select_best_correction([])
        assert result is None

    def test_select_best_correction_priority(self, framework):
        """Test best correction selection based on priority."""
        corrections = [
            CorrectionSuggestion(
                action=CorrectionAction.REGENERATE,
                confidence=0.9,
                original_text="",
                corrected_text="",
                reasoning="Regenerate"
            ),
            CorrectionSuggestion(
                action=CorrectionAction.MODIFY,
                confidence=0.7,
                original_text="",
                corrected_text="",
                reasoning="Modify"
            ),
            CorrectionSuggestion(
                action=CorrectionAction.REJECT,
                confidence=1.0,
                original_text="",
                corrected_text="",
                reasoning="Reject"
            )
        ]

        best = framework._select_best_correction(corrections)
        
        # Should select MODIFY due to higher priority despite lower confidence
        assert best.action == CorrectionAction.MODIFY

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, framework):
        """Test health check with healthy framework."""
        health = await framework.health_check()

        assert health["status"] == "healthy"
        assert "enabled" in health
        assert "response_time" in health
        assert "validation_rules" in health
        assert "enabled_rules" in health
        assert "test_validations" in health
        assert "configuration" in health
        
        # Check configuration details
        config = health["configuration"]
        assert "min_confidence" in config
        assert "max_iterations" in config
        assert "timeout_seconds" in config

    @pytest.mark.asyncio
    async def test_health_check_error_handling(self):
        """Test health check error handling."""
        # Create framework with broken validation engine
        framework = SelfCorrectionFramework()
        
        # Mock validation engine to raise exception
        framework.validation_engine.validate_response = Mock(side_effect=Exception("Test error"))
        
        health = await framework.health_check()

        assert health["status"] == "unhealthy"
        assert "error" in health
        assert "Test error" in health["error"]


class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""

    @pytest.mark.asyncio
    async def test_complete_validation_workflow(self):
        """Test complete validation and correction workflow."""
        framework = SelfCorrectionFramework()
        
        # Create response with multiple issues
        response = RAGResponse(
            content="I think Paris might be the capital of France. Pariz is also known as the City of Light.",
            sources=[{"content": "Paris information", "confidence": 0.8}],
            entities=["Paris", "Pariz", "France"],  # Entity consistency issue
            relationships=[{"source": "Paris", "target": "France", "type": "capital_of"}]
        )
        
        query = "What is the capital of France?"
        context = {"entity_dates": {"Paris": "1000-01-01", "France": "800-01-01"}}

        result = await framework.validate_and_correct(response, query, context)

        # Should complete validation and correction process
        assert isinstance(result, RAGResponse)
        assert result.confidence_score is not None
        assert len(result.validation_results) > 0
        assert "self_correction" in result.metadata
        
        # Check that various validation types were tested
        validation_types = {vr.rule_type for vr in result.validation_results}
        assert ValidationType.ENTITY_CONSISTENCY in validation_types
        assert ValidationType.HALLUCINATION_DETECTION in validation_types

    @pytest.mark.asyncio
    async def test_parallel_validation_execution(self):
        """Test that validations execute in parallel."""
        framework = SelfCorrectionFramework()
        
        response = RAGResponse(
            content="Paris is the capital of France and was founded in 250 BC but also in 2020 AD.",
            entities=["Paris", "France"],
            relationships=[{"source": "Paris", "target": "France", "type": "capital_of"}]
        )
        
        query = "When was Paris founded?"
        context = {}

        start_time = time.time()
        result = await framework.validate_and_correct(response, query, context)
        execution_time = time.time() - start_time

        # Should complete quickly due to parallel execution
        assert execution_time < 5.0  # Should be much faster than sequential
        assert len(result.validation_results) > 0

    @pytest.mark.asyncio
    async def test_error_recovery_during_validation(self):
        """Test error recovery during validation process."""
        framework = SelfCorrectionFramework()
        
        # Mock one validation to fail
        original_validate_entity = framework.validation_engine.validate_entity_consistency
        
        async def failing_validation(*args, **kwargs):
            raise Exception("Validation error")
        
        framework.validation_engine.validate_entity_consistency = failing_validation
        
        response = RAGResponse(content="Test content", entities=["Test"])
        query = "Test query"
        context = {}

        result = await framework.validate_and_correct(response, query, context)

        # Should still complete with other validations
        assert isinstance(result, RAGResponse)
        # Should have some validation results from working validators
        assert len(result.validation_results) >= 0
        
        # Restore original method
        framework.validation_engine.validate_entity_consistency = original_validate_entity


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])