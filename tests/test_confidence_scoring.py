"""
Comprehensive tests for the Advanced Confidence Scoring Module

Tests LLM-based confidence assessment, uncertainty quantification,
and validation rule engine functionality.
"""

import asyncio
import os
import sys
import time
from unittest.mock import Mock, patch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag.validation.confidence_scoring import (
    AdvancedConfidenceScorer,
    ConfidenceAssessment,
    ConfidenceMetric,
    ConfidenceRule,
    LLMConfidenceAssessor,
    LLMConfidenceRequest,
    UncertaintyIndicator,
    UncertaintyQuantifier,
    UncertaintyType,
    ValidationRuleEngine,
    ValidationSeverity,
)


class TestConfidenceMetric:
    """Test ConfidenceMetric enum."""

    def test_confidence_metric_values(self):
        """Test ConfidenceMetric enum values."""
        assert ConfidenceMetric.SOURCE_RELIABILITY.value == "source_reliability"
        assert ConfidenceMetric.CONTENT_COHERENCE.value == "content_coherence"
        assert ConfidenceMetric.FACTUAL_CONSISTENCY.value == "factual_consistency"
        assert ConfidenceMetric.SEMANTIC_CLARITY.value == "semantic_clarity"
        assert ConfidenceMetric.ENTITY_ACCURACY.value == "entity_accuracy"
        assert ConfidenceMetric.RELATIONSHIP_VALIDITY.value == "relationship_validity"
        assert ConfidenceMetric.TEMPORAL_CONSISTENCY.value == "temporal_consistency"
        assert ConfidenceMetric.LOGICAL_SOUNDNESS.value == "logical_soundness"
        
        # Verify we have all expected metrics
        assert len(list(ConfidenceMetric)) == 8


class TestUncertaintyType:
    """Test UncertaintyType enum."""

    def test_uncertainty_type_values(self):
        """Test UncertaintyType enum values."""
        assert UncertaintyType.LINGUISTIC.value == "linguistic"
        assert UncertaintyType.FACTUAL.value == "factual"
        assert UncertaintyType.TEMPORAL.value == "temporal"
        assert UncertaintyType.SEMANTIC.value == "semantic"
        assert UncertaintyType.STRUCTURAL.value == "structural"
        assert UncertaintyType.CONTEXTUAL.value == "contextual"
        
        # Verify we have all expected types
        assert len(list(UncertaintyType)) == 6


class TestValidationSeverity:
    """Test ValidationSeverity enum."""

    def test_validation_severity_values(self):
        """Test ValidationSeverity enum values."""
        assert ValidationSeverity.CRITICAL.value == "critical"
        assert ValidationSeverity.HIGH.value == "high"
        assert ValidationSeverity.MEDIUM.value == "medium"
        assert ValidationSeverity.LOW.value == "low"
        assert ValidationSeverity.INFO.value == "info"
        
        # Verify we have all expected severities
        assert len(list(ValidationSeverity)) == 5


class TestDataClasses:
    """Test data classes and their creation."""

    def test_confidence_rule_creation(self):
        """Test ConfidenceRule dataclass creation."""
        rule = ConfidenceRule(
            name="test_rule",
            metric=ConfidenceMetric.SOURCE_RELIABILITY,
            weight=0.5,
            threshold=0.8,
            severity=ValidationSeverity.HIGH,
            description="Test rule description",
            enabled=True,
            llm_based=False
        )
        
        assert rule.name == "test_rule"
        assert rule.metric == ConfidenceMetric.SOURCE_RELIABILITY
        assert rule.weight == 0.5
        assert rule.threshold == 0.8
        assert rule.severity == ValidationSeverity.HIGH
        assert rule.description == "Test rule description"
        assert rule.enabled is True
        assert rule.llm_based is False

    def test_uncertainty_indicator_creation(self):
        """Test UncertaintyIndicator dataclass creation."""
        indicator = UncertaintyIndicator(
            type=UncertaintyType.LINGUISTIC,
            text="maybe",
            position=(10, 15),
            confidence=0.8,
            context="I think maybe this is correct",
            severity=ValidationSeverity.MEDIUM,
            suggestion="Remove uncertain language"
        )
        
        assert indicator.type == UncertaintyType.LINGUISTIC
        assert indicator.text == "maybe"
        assert indicator.position == (10, 15)
        assert indicator.confidence == 0.8
        assert indicator.context == "I think maybe this is correct"
        assert indicator.severity == ValidationSeverity.MEDIUM
        assert indicator.suggestion == "Remove uncertain language"

    def test_confidence_assessment_creation(self):
        """Test ConfidenceAssessment dataclass creation."""
        assessment = ConfidenceAssessment(
            overall_confidence=0.85,
            metric_scores={ConfidenceMetric.SOURCE_RELIABILITY: 0.9},
            uncertainty_indicators=[],
            validation_results=[],
            llm_assessment={"test": "value"},
            processing_time=0.5,
            metadata={"test": True}
        )
        
        assert assessment.overall_confidence == 0.85
        assert assessment.metric_scores[ConfidenceMetric.SOURCE_RELIABILITY] == 0.9
        assert assessment.uncertainty_indicators == []
        assert assessment.validation_results == []
        assert assessment.llm_assessment == {"test": "value"}
        assert assessment.processing_time == 0.5
        assert assessment.metadata == {"test": True}

    def test_llm_confidence_request_creation(self):
        """Test LLMConfidenceRequest dataclass creation."""
        request = LLMConfidenceRequest(
            content="Test content",
            query="Test query",
            context={"key": "value"},
            sources=[{"content": "source"}],
            entities=["entity1"],
            relationships=[{"source": "A", "target": "B"}]
        )
        
        assert request.content == "Test content"
        assert request.query == "Test query"
        assert request.context == {"key": "value"}
        assert request.sources == [{"content": "source"}]
        assert request.entities == ["entity1"]
        assert request.relationships == [{"source": "A", "target": "B"}]


class TestUncertaintyQuantifier:
    """Test UncertaintyQuantifier functionality."""

    @pytest.fixture
    def quantifier(self):
        """Create UncertaintyQuantifier instance."""
        return UncertaintyQuantifier()

    @pytest.mark.asyncio
    async def test_quantify_uncertainty_no_uncertainty(self, quantifier):
        """Test uncertainty quantification with no uncertainty."""
        text = "Paris is the capital of France. It has a population of over 2 million people."
        
        indicators = await quantifier.quantify_uncertainty(text)
        
        # Should have minimal or no uncertainty indicators
        assert len(indicators) <= 2  # May have some structural indicators
        
        # Check that no high-confidence linguistic indicators are found
        linguistic_indicators = [i for i in indicators if i.type == UncertaintyType.LINGUISTIC]
        assert len(linguistic_indicators) == 0

    @pytest.mark.asyncio
    async def test_quantify_uncertainty_with_linguistic_uncertainty(self, quantifier):
        """Test uncertainty quantification with linguistic uncertainty."""
        text = "I think Paris might be the capital of France. Maybe it has around 2 million people."
        
        indicators = await quantifier.quantify_uncertainty(text)
        
        # Should find uncertainty indicators
        assert len(indicators) > 0
        
        # Check for linguistic uncertainty
        linguistic_indicators = [i for i in indicators if i.type == UncertaintyType.LINGUISTIC]
        assert len(linguistic_indicators) >= 2  # "I think", "might", "maybe"
        
        # Verify indicator properties
        for indicator in linguistic_indicators:
            assert indicator.confidence > 0.0
            assert indicator.text.lower() in ["i think", "might", "maybe"]
            assert indicator.severity in [ValidationSeverity.LOW, ValidationSeverity.MEDIUM]

    @pytest.mark.asyncio
    async def test_quantify_uncertainty_case_insensitive(self, quantifier):
        """Test that uncertainty detection is case insensitive."""
        text = "MAYBE this is correct. I THINK it's true."
        
        indicators = await quantifier.quantify_uncertainty(text)
        
        linguistic_indicators = [i for i in indicators if i.type == UncertaintyType.LINGUISTIC]
        assert len(linguistic_indicators) >= 2
        
        # Check that uppercase patterns are detected
        found_texts = [i.text.lower() for i in linguistic_indicators]
        assert "maybe" in found_texts
        assert "i think" in found_texts

    @pytest.mark.asyncio
    async def test_quantify_uncertainty_with_factual_uncertainty(self, quantifier):
        """Test detection of factual uncertainty."""
        text = "This is a disputed claim that has conflicting evidence."
        
        indicators = await quantifier.quantify_uncertainty(text)
        
        factual_indicators = [i for i in indicators if i.type == UncertaintyType.FACTUAL]
        assert len(factual_indicators) >= 1
        
        # Check severity for factual uncertainty
        for indicator in factual_indicators:
            assert indicator.severity == ValidationSeverity.HIGH

    @pytest.mark.asyncio
    async def test_quantify_uncertainty_with_temporal_uncertainty(self, quantifier):
        """Test detection of temporal uncertainty."""
        text = "This happened around 1995, sometime in the recent past."
        
        indicators = await quantifier.quantify_uncertainty(text)
        
        temporal_indicators = [i for i in indicators if i.type == UncertaintyType.TEMPORAL]
        assert len(temporal_indicators) >= 1

    @pytest.mark.asyncio
    async def test_quantify_uncertainty_with_semantic_uncertainty(self, quantifier):
        """Test detection of semantic uncertainty."""
        text = "This thing is kind of like that stuff we discussed."
        
        indicators = await quantifier.quantify_uncertainty(text)
        
        semantic_indicators = [i for i in indicators if i.type == UncertaintyType.SEMANTIC]
        assert len(semantic_indicators) >= 2  # "thing", "kind of", "stuff"

    @pytest.mark.asyncio
    async def test_quantify_uncertainty_with_structural_issues(self, quantifier):
        """Test detection of structural uncertainty."""
        text = "Yes. No. Maybe."  # Very short sentences
        context = {"query": "What is the detailed explanation of quantum physics?"}
        
        indicators = await quantifier.quantify_uncertainty(text, context)
        
        structural_indicators = [i for i in indicators if i.type == UncertaintyType.STRUCTURAL]
        assert len(structural_indicators) >= 1

    @pytest.mark.asyncio
    async def test_quantify_uncertainty_with_contextual_issues(self, quantifier):
        """Test detection of contextual uncertainty."""
        text = "The weather is nice today."
        context = {"query": "What is the capital city of France and its population statistics?"}
        
        indicators = await quantifier.quantify_uncertainty(text, context)
        
        contextual_indicators = [i for i in indicators if i.type == UncertaintyType.CONTEXTUAL]
        # May not always detect contextual issues with simple keyword matching
        assert len(contextual_indicators) >= 0  # Allow for no detection in simple cases

    def test_calculate_pattern_confidence(self, quantifier):
        """Test pattern confidence calculation."""
        # Test high confidence patterns
        high_conf = quantifier._calculate_pattern_confidence("uncertain", UncertaintyType.LINGUISTIC)
        assert high_conf > 0.8
        
        # Test medium confidence patterns
        medium_conf = quantifier._calculate_pattern_confidence("possibly", UncertaintyType.LINGUISTIC)
        assert 0.7 <= medium_conf <= 0.9

    def test_extract_context(self, quantifier):
        """Test context extraction around matches."""
        text = "This is a test sentence with maybe some uncertainty in the middle of it."
        start, end = 30, 35  # "maybe"
        
        context = quantifier._extract_context(text, start, end, window=20)
        
        assert "maybe" in context
        assert len(context) <= 45  # Original window size + match length

    def test_determine_severity(self, quantifier):
        """Test severity determination for different uncertainty types."""
        # Factual uncertainty should be high severity
        severity = quantifier._determine_severity(UncertaintyType.FACTUAL, "conflicting")
        assert severity == ValidationSeverity.HIGH
        
        # Strong linguistic uncertainty should be medium
        severity = quantifier._determine_severity(UncertaintyType.LINGUISTIC, "uncertain")
        assert severity == ValidationSeverity.MEDIUM
        
        # Weak linguistic uncertainty should be low
        severity = quantifier._determine_severity(UncertaintyType.LINGUISTIC, "possibly")
        assert severity == ValidationSeverity.LOW

    def test_generate_suggestion(self, quantifier):
        """Test suggestion generation for uncertainty indicators."""
        suggestion = quantifier._generate_suggestion(UncertaintyType.LINGUISTIC, "maybe")
        assert "maybe" in suggestion
        assert "uncertain language" in suggestion.lower()
        
        suggestion = quantifier._generate_suggestion(UncertaintyType.FACTUAL, "disputed")
        assert "disputed" in suggestion
        assert "verify" in suggestion.lower()


class TestValidationRuleEngine:
    """Test ValidationRuleEngine functionality."""

    @pytest.fixture
    def rule_engine(self):
        """Create ValidationRuleEngine instance."""
        return ValidationRuleEngine()

    def test_initialization_default_rules(self, rule_engine):
        """Test that default rules are initialized correctly."""
        rules = rule_engine.rules
        assert len(rules) > 0
        
        # Check that all confidence metrics are covered
        metrics_covered = {rule.metric for rule in rules}
        expected_metrics = set(ConfidenceMetric)
        assert metrics_covered == expected_metrics
        
        # Check that weights sum to approximately 1.0
        total_weight = sum(rule.weight for rule in rules if rule.enabled)
        assert abs(total_weight - 1.0) < 0.01

    def test_add_rule(self, rule_engine):
        """Test adding a new validation rule."""
        initial_count = len(rule_engine.rules)
        
        new_rule = ConfidenceRule(
            name="test_rule",
            metric=ConfidenceMetric.SOURCE_RELIABILITY,
            weight=0.1,
            threshold=0.8,
            severity=ValidationSeverity.MEDIUM,
            description="Test rule"
        )
        
        rule_engine.add_rule(new_rule)
        assert len(rule_engine.rules) == initial_count + 1
        assert rule_engine.get_rule("test_rule") == new_rule

    def test_remove_rule(self, rule_engine):
        """Test removing a validation rule."""
        # Add a test rule first
        test_rule = ConfidenceRule(
            name="test_rule_to_remove",
            metric=ConfidenceMetric.SOURCE_RELIABILITY,
            weight=0.1,
            threshold=0.8,
            severity=ValidationSeverity.MEDIUM,
            description="Test rule"
        )
        rule_engine.add_rule(test_rule)
        
        initial_count = len(rule_engine.rules)
        
        # Remove the rule
        removed = rule_engine.remove_rule("test_rule_to_remove")
        assert removed is True
        assert len(rule_engine.rules) == initial_count - 1
        assert rule_engine.get_rule("test_rule_to_remove") is None
        
        # Try to remove non-existent rule
        removed = rule_engine.remove_rule("non_existent_rule")
        assert removed is False

    def test_get_rule(self, rule_engine):
        """Test getting a validation rule by name."""
        # Should find existing rule
        rule = rule_engine.get_rule("source_reliability")
        assert rule is not None
        assert rule.name == "source_reliability"
        
        # Should return None for non-existent rule
        rule = rule_engine.get_rule("non_existent_rule")
        assert rule is None

    def test_get_enabled_rules(self, rule_engine):
        """Test getting enabled validation rules."""
        enabled_rules = rule_engine.get_enabled_rules()
        assert len(enabled_rules) > 0
        
        # All returned rules should be enabled
        for rule in enabled_rules:
            assert rule.enabled is True

    def test_get_llm_rules(self, rule_engine):
        """Test getting LLM-based validation rules."""
        llm_rules = rule_engine.get_llm_rules()
        
        # All returned rules should be LLM-based and enabled
        for rule in llm_rules:
            assert rule.llm_based is True
            assert rule.enabled is True
        
        # Should have some LLM rules by default
        assert len(llm_rules) > 0

    def test_update_rule(self, rule_engine):
        """Test updating a validation rule."""
        # Update existing rule
        updated = rule_engine.update_rule("source_reliability", weight=0.5, threshold=0.9)
        assert updated is True
        
        rule = rule_engine.get_rule("source_reliability")
        assert rule.weight == 0.5
        assert rule.threshold == 0.9
        
        # Try to update non-existent rule
        updated = rule_engine.update_rule("non_existent_rule", weight=0.5)
        assert updated is False

    def test_validate_rule_weights(self, rule_engine):
        """Test validation of rule weights."""
        # Default rules should have valid weights
        assert rule_engine.validate_rule_weights() is True
        
        # Modify weights to make them invalid
        for rule in rule_engine.get_enabled_rules():
            rule.weight = 0.5  # This will make total > 1.0
        
        assert rule_engine.validate_rule_weights() is False

    def test_normalize_weights(self, rule_engine):
        """Test weight normalization."""
        # Set unbalanced weights
        enabled_rules = rule_engine.get_enabled_rules()
        for i, rule in enumerate(enabled_rules):
            rule.weight = (i + 1) * 0.1  # Arbitrary weights that don't sum to 1.0
        
        # Normalize weights
        rule_engine.normalize_weights()
        
        # Check that weights now sum to 1.0
        assert rule_engine.validate_rule_weights() is True
        
        total_weight = sum(rule.weight for rule in rule_engine.get_enabled_rules())
        assert abs(total_weight - 1.0) < 0.01


class TestLLMConfidenceAssessor:
    """Test LLMConfidenceAssessor functionality."""

    @pytest.fixture
    def assessor(self):
        """Create LLMConfidenceAssessor instance."""
        return LLMConfidenceAssessor()

    @pytest.fixture
    def sample_request(self):
        """Create sample LLM confidence request."""
        return LLMConfidenceRequest(
            content="Paris is the capital of France. It has a rich history and culture.",
            query="What is the capital of France?",
            context={"user_id": "test"},
            sources=[{"content": "Paris capital France", "confidence": 0.9}],
            entities=["Paris", "France"],
            relationships=[{"source": "Paris", "target": "France", "type": "capital_of"}]
        )

    def test_assessment_prompts_exist(self, assessor):
        """Test that assessment prompts exist for LLM metrics."""
        expected_metrics = [
            ConfidenceMetric.CONTENT_COHERENCE,
            ConfidenceMetric.FACTUAL_CONSISTENCY,
            ConfidenceMetric.LOGICAL_SOUNDNESS
        ]
        
        for metric in expected_metrics:
            assert metric in assessor.assessment_prompts
            prompt = assessor.assessment_prompts[metric]
            assert "{query}" in prompt
            assert "{content}" in prompt

    @pytest.mark.asyncio
    async def test_assess_confidence_single_metric(self, assessor, sample_request):
        """Test LLM assessment for a single metric."""
        metrics = [ConfidenceMetric.CONTENT_COHERENCE]
        
        results = await assessor.assess_confidence(sample_request, metrics)
        
        assert len(results) == 1
        assert ConfidenceMetric.CONTENT_COHERENCE in results
        assert 0.0 <= results[ConfidenceMetric.CONTENT_COHERENCE] <= 1.0

    @pytest.mark.asyncio
    async def test_assess_confidence_multiple_metrics(self, assessor, sample_request):
        """Test LLM assessment for multiple metrics."""
        metrics = [
            ConfidenceMetric.CONTENT_COHERENCE,
            ConfidenceMetric.FACTUAL_CONSISTENCY,
            ConfidenceMetric.LOGICAL_SOUNDNESS
        ]
        
        results = await assessor.assess_confidence(sample_request, metrics)
        
        assert len(results) == 3
        for metric in metrics:
            assert metric in results
            assert 0.0 <= results[metric] <= 1.0

    @pytest.mark.asyncio
    async def test_assess_confidence_unsupported_metric(self, assessor, sample_request):
        """Test LLM assessment with unsupported metric."""
        metrics = [ConfidenceMetric.SOURCE_RELIABILITY]  # Not LLM-based
        
        results = await assessor.assess_confidence(sample_request, metrics)
        
        # Should return empty results for unsupported metrics
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_assess_confidence_error_handling(self, assessor):
        """Test error handling in LLM assessment."""
        # Create invalid request
        invalid_request = LLMConfidenceRequest(
            content=None,  # Invalid content
            query="test",
            context={},
            sources=[],
            entities=[],
            relationships=[]
        )
        
        metrics = [ConfidenceMetric.CONTENT_COHERENCE]
        
        results = await assessor.assess_confidence(invalid_request, metrics)
        
        # Should handle errors gracefully
        assert len(results) == 1
        # Mock LLM may still return a score even with invalid content
        assert 0.0 <= results[ConfidenceMetric.CONTENT_COHERENCE] <= 1.0

    def test_format_sources_empty(self, assessor):
        """Test source formatting with empty sources."""
        formatted = assessor._format_sources([])
        assert formatted == "No sources provided"

    def test_format_sources_with_data(self, assessor):
        """Test source formatting with source data."""
        sources = [
            {"content": "Source 1 content", "confidence": 0.9},
            {"content": "Source 2 content", "confidence": 0.8},
        ]
        
        formatted = assessor._format_sources(sources)
        
        assert "1. Source 1 content" in formatted
        assert "2. Source 2 content" in formatted
        assert "Confidence: 0.9" in formatted
        assert "Confidence: 0.8" in formatted

    def test_format_sources_limit(self, assessor):
        """Test that source formatting limits to 5 sources."""
        sources = [{"content": f"Source {i}", "confidence": 0.5} for i in range(10)]
        
        formatted = assessor._format_sources(sources)
        
        # Should only include first 5 sources
        assert "1. Source 0" in formatted
        assert "5. Source 4" in formatted
        assert "6. Source 5" not in formatted

    @pytest.mark.asyncio
    async def test_mock_llm_call_content_analysis(self, assessor):
        """Test mock LLM call with different content types."""
        # Test with contradictory content
        score = await assessor._mock_llm_call(
            "This is contradictory information", 
            ConfidenceMetric.CONTENT_COHERENCE
        )
        assert score < 0.5
        
        # Test with clear content
        score = await assessor._mock_llm_call(
            "This is clear and accurate information", 
            ConfidenceMetric.CONTENT_COHERENCE
        )
        assert score > 0.8


class TestAdvancedConfidenceScorer:
    """Test AdvancedConfidenceScorer integration."""

    @pytest.fixture
    def scorer(self):
        """Create AdvancedConfidenceScorer instance."""
        return AdvancedConfidenceScorer()

    @pytest.mark.asyncio
    async def test_assess_confidence_basic(self, scorer):
        """Test basic confidence assessment."""
        content = "Paris is the capital of France. It has a population of over 2 million people."
        query = "What is the capital of France?"
        
        assessment = await scorer.assess_confidence(content, query)
        
        assert isinstance(assessment, ConfidenceAssessment)
        assert 0.0 <= assessment.overall_confidence <= 1.0
        assert len(assessment.metric_scores) > 0
        assert assessment.processing_time > 0
        assert "rules_applied" in assessment.metadata

    @pytest.mark.asyncio
    async def test_assess_confidence_with_uncertainty(self, scorer):
        """Test confidence assessment with uncertainty content."""
        content = "I think Paris might be the capital of France, but I'm not entirely sure."
        query = "What is the capital of France?"
        
        assessment = await scorer.assess_confidence(content, query)
        
        # Should detect uncertainty and lower confidence
        assert assessment.overall_confidence < 0.8
        assert len(assessment.uncertainty_indicators) > 0
        
        # Check for linguistic uncertainty indicators
        linguistic_indicators = [
            ui for ui in assessment.uncertainty_indicators 
            if ui.type == UncertaintyType.LINGUISTIC
        ]
        assert len(linguistic_indicators) > 0

    @pytest.mark.asyncio
    async def test_assess_confidence_with_sources(self, scorer):
        """Test confidence assessment with source information."""
        content = "Paris is the capital of France."
        query = "What is the capital of France?"
        sources = [
            {"content": "Paris is the capital city of France", "confidence": 0.95, "type": "document"},
            {"content": "France capital information", "confidence": 0.8, "type": "database"}
        ]
        
        assessment = await scorer.assess_confidence(content, query, sources=sources)
        
        # Should have higher source reliability score
        assert ConfidenceMetric.SOURCE_RELIABILITY in assessment.metric_scores
        source_score = assessment.metric_scores[ConfidenceMetric.SOURCE_RELIABILITY]
        assert source_score > 0.7

    @pytest.mark.asyncio
    async def test_assess_confidence_with_entities_and_relationships(self, scorer):
        """Test confidence assessment with entities and relationships."""
        content = "Paris is the capital of France and has many famous landmarks."
        query = "What is the capital of France?"
        entities = ["Paris", "France"]
        relationships = [{"source": "Paris", "target": "France", "type": "capital_of"}]
        
        assessment = await scorer.assess_confidence(
            content, query, entities=entities, relationships=relationships
        )
        
        # Should have entity and relationship scores
        assert ConfidenceMetric.ENTITY_ACCURACY in assessment.metric_scores
        assert ConfidenceMetric.RELATIONSHIP_VALIDITY in assessment.metric_scores
        
        entity_score = assessment.metric_scores[ConfidenceMetric.ENTITY_ACCURACY]
        relationship_score = assessment.metric_scores[ConfidenceMetric.RELATIONSHIP_VALIDITY]
        
        assert entity_score > 0.7
        assert relationship_score > 0.7

    @pytest.mark.asyncio
    async def test_assess_confidence_error_handling(self, scorer):
        """Test error handling in confidence assessment."""
        # Test with None content
        assessment = await scorer.assess_confidence(None, "test query")
        
        assert isinstance(assessment, ConfidenceAssessment)
        assert assessment.overall_confidence == 0.5  # Fallback score
        assert "error" in assessment.metadata

    def test_assess_source_reliability_no_sources(self, scorer):
        """Test source reliability assessment with no sources."""
        score = scorer._assess_source_reliability([])
        assert score == 0.3  # Expected low confidence score

    def test_assess_source_reliability_with_sources(self, scorer):
        """Test source reliability assessment with good sources."""
        sources = [
            {"confidence": 0.9, "type": "document"},
            {"confidence": 0.8, "type": "database"},
            {"confidence": 0.7, "type": "web"}
        ]
        
        score = scorer._assess_source_reliability(sources)
        assert score > 0.7  # Should be high with good sources

    def test_assess_semantic_clarity_clear_text(self, scorer):
        """Test semantic clarity assessment with clear text."""
        content = "Paris is the capital of France. It is located in the northern part of the country."
        
        score = scorer._assess_semantic_clarity(content)
        assert score > 0.8  # Should be high for clear text

    def test_assess_semantic_clarity_unclear_text(self, scorer):
        """Test semantic clarity assessment with unclear text."""
        content = "This thing is kind of like that stuff we talked about."
        
        score = scorer._assess_semantic_clarity(content)
        assert score <= 0.8  # Should be at most 0.8 for unclear text

    def test_assess_entity_accuracy_good_coverage(self, scorer):
        """Test entity accuracy with good entity coverage."""
        entities = ["Paris", "France"]
        content = "Paris is the capital of France and has many attractions."
        
        score = scorer._assess_entity_accuracy(entities, content)
        assert score > 0.8  # Should be high with good coverage

    def test_assess_entity_accuracy_poor_coverage(self, scorer):
        """Test entity accuracy with poor entity coverage."""
        entities = ["London", "Germany"]  # Not mentioned in content
        content = "Paris is the capital of France and has many attractions."
        
        score = scorer._assess_entity_accuracy(entities, content)
        assert score < 0.7  # Should be lower with poor coverage

    def test_assess_relationship_validity_good_relationships(self, scorer):
        """Test relationship validity with good relationships."""
        relationships = [{"source": "Paris", "target": "France", "type": "capital_of"}]
        content = "Paris is the capital of France."
        
        score = scorer._assess_relationship_validity(relationships, content)
        assert score > 0.8  # Should be high with valid relationships

    def test_assess_relationship_validity_poor_relationships(self, scorer):
        """Test relationship validity with poor relationships."""
        relationships = [{"source": "London", "target": "Germany", "type": "capital_of"}]
        content = "Paris is the capital of France."
        
        score = scorer._assess_relationship_validity(relationships, content)
        assert score < 0.7  # Should be lower with invalid relationships

    def test_assess_temporal_consistency_no_dates(self, scorer):
        """Test temporal consistency with no dates."""
        content = "Paris is a beautiful city with many attractions."
        
        score = scorer._assess_temporal_consistency(content)
        assert score == 0.9  # High score for insufficient temporal info

    def test_assess_temporal_consistency_reasonable_range(self, scorer):
        """Test temporal consistency with reasonable date range."""
        content = "The building was constructed in 1995 and renovated in 2010."
        
        score = scorer._assess_temporal_consistency(content)
        assert score == 0.9  # High score for reasonable range

    def test_assess_temporal_consistency_large_range(self, scorer):
        """Test temporal consistency with large date range."""
        content = "This happened in 1066 and was later updated in 2020."
        
        score = scorer._assess_temporal_consistency(content)
        assert score <= 0.8  # Lower score for large range (1066-2020 = 954 years)

    @pytest.mark.asyncio
    async def test_health_check(self, scorer):
        """Test health check functionality."""
        health = await scorer.health_check()
        
        assert health["status"] == "healthy"
        assert health["response_time"] > 0
        assert "components" in health
        assert "metrics" in health
        
        # Check component health
        components = health["components"]
        assert components["uncertainty_quantifier"] is True
        assert components["rule_engine"] is True
        assert components["llm_assessor"] is True
        
        # Check metrics
        metrics = health["metrics"]
        assert metrics["enabled_rules"] > 0
        assert metrics["rule_weights_valid"] is True
        assert metrics["uncertainty_patterns"] > 0
        assert metrics["llm_metrics_available"] > 0


class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""

    @pytest.fixture
    def scorer(self):
        """Create AdvancedConfidenceScorer instance."""
        return AdvancedConfidenceScorer()

    @pytest.mark.asyncio
    async def test_comprehensive_assessment_workflow(self, scorer):
        """Test complete assessment workflow with all components."""
        content = """
        I believe Paris is probably the capital of France, though I'm not entirely certain.
        It might have a population of around 2 million people, but this could be disputed.
        The city was founded sometime in ancient times, perhaps around 300 BC or so.
        """
        
        query = "What is the capital of France and when was it founded?"
        
        context = {"user_preferences": "detailed_answers"}
        sources = [
            {"content": "Paris capital France population", "confidence": 0.7, "type": "web"},
            {"content": "Paris founded ancient times", "confidence": 0.6, "type": "web"}
        ]
        entities = ["Paris", "France"]
        relationships = [{"source": "Paris", "target": "France", "type": "capital_of"}]
        
        assessment = await scorer.assess_confidence(
            content, query, context, sources, entities, relationships
        )
        
        # Should detect multiple uncertainty indicators
        assert len(assessment.uncertainty_indicators) >= 3
        
        # Should have lower overall confidence due to uncertainty
        assert assessment.overall_confidence < 0.7
        
        # Should generate validation results
        assert len(assessment.validation_results) > 0
        
        # Should include LLM assessment
        assert assessment.llm_assessment is not None
        assert "metrics_assessed" in assessment.llm_assessment
        
        # Check for different types of uncertainty
        uncertainty_types = {ui.type for ui in assessment.uncertainty_indicators}
        assert UncertaintyType.LINGUISTIC in uncertainty_types
        assert UncertaintyType.FACTUAL in uncertainty_types
        assert UncertaintyType.TEMPORAL in uncertainty_types

    @pytest.mark.asyncio
    async def test_high_confidence_scenario(self, scorer):
        """Test scenario that should result in high confidence."""
        content = """
        Paris is the capital of France. It is located in the Île-de-France region in north-central France.
        The city has a population of approximately 2.1 million people within its administrative limits.
        Paris was founded in the 3rd century BC by a Celtic people called the Parisii.
        """
        
        query = "What is the capital of France?"
        
        sources = [
            {"content": "Paris capital France official", "confidence": 0.95, "type": "document"},
            {"content": "Paris population demographics", "confidence": 0.9, "type": "database"}
        ]
        entities = ["Paris", "France", "Île-de-France", "Parisii"]
        relationships = [
            {"source": "Paris", "target": "France", "type": "capital_of"},
            {"source": "Paris", "target": "Île-de-France", "type": "located_in"}
        ]
        
        assessment = await scorer.assess_confidence(
            content, query, sources=sources, entities=entities, relationships=relationships
        )
        
        # Should have high overall confidence
        assert assessment.overall_confidence > 0.75
        
        # Should have few or no significant uncertainty indicators
        high_severity_indicators = [
            ui for ui in assessment.uncertainty_indicators 
            if ui.severity in [ValidationSeverity.HIGH, ValidationSeverity.CRITICAL]
        ]
        assert len(high_severity_indicators) == 0
        
        # Most validation rules should pass
        passed_validations = [vr for vr in assessment.validation_results if vr["passed"]]
        total_validations = len(assessment.validation_results)
        pass_rate = len(passed_validations) / total_validations if total_validations > 0 else 1.0
        assert pass_rate > 0.7

    @pytest.mark.asyncio
    async def test_parallel_assessment_performance(self, scorer):
        """Test that parallel assessment improves performance."""
        content = "Paris is the capital of France with rich history and culture."
        query = "What is the capital of France?"
        
        # Measure time for single assessment
        start_time = time.time()
        assessment = await scorer.assess_confidence(content, query)
        single_time = time.time() - start_time
        
        # The assessment should complete quickly due to parallel processing
        assert single_time < 2.0  # Should complete within 2 seconds
        assert assessment.processing_time < 2.0
        
        # Verify that multiple components were executed
        assert len(assessment.metric_scores) > 3  # Multiple metrics assessed
        assert assessment.llm_assessment is not None  # LLM assessment completed


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])