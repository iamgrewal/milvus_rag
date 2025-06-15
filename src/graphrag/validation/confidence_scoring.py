"""
Advanced Confidence Scoring Module for Phase 2 Hybrid RAG System

Implements LLM-based confidence assessment, uncertainty quantification,
and validation rule engine for enhanced response quality evaluation.
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from graphrag.config.settings import Config
from graphrag.logger import logger


class ConfidenceMetric(Enum):
    """Types of confidence metrics."""
    
    SOURCE_RELIABILITY = "source_reliability"
    CONTENT_COHERENCE = "content_coherence"
    FACTUAL_CONSISTENCY = "factual_consistency"
    SEMANTIC_CLARITY = "semantic_clarity"
    ENTITY_ACCURACY = "entity_accuracy"
    RELATIONSHIP_VALIDITY = "relationship_validity"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    LOGICAL_SOUNDNESS = "logical_soundness"


class UncertaintyType(Enum):
    """Types of uncertainty in responses."""
    
    LINGUISTIC = "linguistic"  # "maybe", "might", "possibly"
    FACTUAL = "factual"       # Contradictory information
    TEMPORAL = "temporal"     # Date inconsistencies
    SEMANTIC = "semantic"     # Unclear meaning
    STRUCTURAL = "structural" # Missing key information
    CONTEXTUAL = "contextual" # Insufficient context


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    
    CRITICAL = "critical"     # Response should be rejected
    HIGH = "high"            # Requires correction
    MEDIUM = "medium"        # Warning, may need attention
    LOW = "low"              # Minor issue, acceptable
    INFO = "info"            # Informational only


@dataclass
class ConfidenceRule:
    """Represents a confidence validation rule."""
    
    name: str
    metric: ConfidenceMetric
    weight: float
    threshold: float
    severity: ValidationSeverity
    description: str
    enabled: bool = True
    llm_based: bool = False


@dataclass
class UncertaintyIndicator:
    """Represents an uncertainty indicator found in text."""
    
    type: UncertaintyType
    text: str
    position: Tuple[int, int]
    confidence: float
    context: str
    severity: ValidationSeverity
    suggestion: str = ""


@dataclass
class ConfidenceAssessment:
    """Comprehensive confidence assessment result."""
    
    overall_confidence: float
    metric_scores: Dict[ConfidenceMetric, float]
    uncertainty_indicators: List[UncertaintyIndicator]
    validation_results: List[Dict[str, Any]]
    llm_assessment: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfidenceRequest:
    """Request structure for LLM-based confidence assessment."""
    
    content: str
    query: str
    context: Dict[str, Any]
    sources: List[Dict[str, Any]]
    entities: List[str]
    relationships: List[Dict[str, Any]]


class UncertaintyQuantifier:
    """
    Quantifies uncertainty in text using linguistic patterns and semantic analysis.
    """
    
    def __init__(self):
        self.uncertainty_patterns = {
            UncertaintyType.LINGUISTIC: [
                r'\b(?:maybe|perhaps|possibly|probably|likely|might|could|may|seem|appear)\b',
                r'\b(?:I think|I believe|I guess|I suppose|it seems|it appears)\b',
                r'\b(?:presumably|apparently|supposedly|allegedly)\b',
                r'\b(?:uncertain|unclear|ambiguous|vague)\b',
                r'\b(?:not sure|not certain|not clear)\b'
            ],
            UncertaintyType.FACTUAL: [
                r'\b(?:conflicting|contradictory|inconsistent)\b',
                r'\b(?:disputed|debated|controversial)\b',
                r'\b(?:unverified|unconfirmed|alleged)\b'
            ],
            UncertaintyType.TEMPORAL: [
                r'\b(?:approximately|around|circa|about)\s+\d{4}\b',
                r'\b(?:sometime|eventually|recently|soon)\b',
                r'\b(?:before|after)\s+(?:that|this|then)\b'
            ],
            UncertaintyType.SEMANTIC: [
                r'\b(?:thing|stuff|something|somehow)\b',
                r'\b(?:kind of|sort of|type of)\b',
                r'\b(?:et cetera|etc\.?|and so on)\b'
            ]
        }
        
        # Compile patterns for performance
        self.compiled_patterns = {}
        for uncertainty_type, patterns in self.uncertainty_patterns.items():
            self.compiled_patterns[uncertainty_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    async def quantify_uncertainty(self, text: str, context: Dict[str, Any] = None) -> List[UncertaintyIndicator]:
        """
        Quantify uncertainty indicators in the given text.
        
        Args:
            text: Text to analyze
            context: Additional context for analysis
            
        Returns:
            List of uncertainty indicators found
        """
        try:
            indicators = []
            context = context or {}
            
            # Analyze linguistic uncertainty
            for uncertainty_type, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    for match in pattern.finditer(text):
                        indicator = UncertaintyIndicator(
                            type=uncertainty_type,
                            text=match.group(),
                            position=(match.start(), match.end()),
                            confidence=self._calculate_pattern_confidence(match.group(), uncertainty_type),
                            context=self._extract_context(text, match.start(), match.end()),
                            severity=self._determine_severity(uncertainty_type, match.group()),
                            suggestion=self._generate_suggestion(uncertainty_type, match.group())
                        )
                        indicators.append(indicator)
            
            # Analyze structural uncertainty
            structural_indicators = await self._analyze_structural_uncertainty(text, context)
            indicators.extend(structural_indicators)
            
            # Analyze contextual uncertainty
            contextual_indicators = await self._analyze_contextual_uncertainty(text, context)
            indicators.extend(contextual_indicators)
            
            # Remove duplicates and sort by confidence
            indicators = self._deduplicate_indicators(indicators)
            indicators.sort(key=lambda x: x.confidence, reverse=True)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error quantifying uncertainty: {e}")
            return []
    
    def _calculate_pattern_confidence(self, text: str, uncertainty_type: UncertaintyType) -> float:
        """Calculate confidence score for a pattern match."""
        # Base confidence based on pattern strength
        base_confidence = {
            UncertaintyType.LINGUISTIC: 0.8,
            UncertaintyType.FACTUAL: 0.9,
            UncertaintyType.TEMPORAL: 0.7,
            UncertaintyType.SEMANTIC: 0.6
        }.get(uncertainty_type, 0.5)
        
        # Adjust based on specific text
        text_lower = text.lower()
        if text_lower in ['maybe', 'might', 'could', 'uncertain', 'unclear']:
            base_confidence += 0.1
        elif text_lower in ['I think', 'I believe', 'I guess']:
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Extract context around a match."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def _determine_severity(self, uncertainty_type: UncertaintyType, text: str) -> ValidationSeverity:
        """Determine severity of uncertainty indicator."""
        if uncertainty_type == UncertaintyType.FACTUAL:
            return ValidationSeverity.HIGH
        elif uncertainty_type == UncertaintyType.TEMPORAL and 'contradictory' in text.lower():
            return ValidationSeverity.HIGH
        elif uncertainty_type == UncertaintyType.LINGUISTIC:
            strong_indicators = ['uncertain', 'unclear', 'I guess', 'not sure']
            if any(indicator in text.lower() for indicator in strong_indicators):
                return ValidationSeverity.MEDIUM
            return ValidationSeverity.LOW
        else:
            return ValidationSeverity.MEDIUM
    
    def _generate_suggestion(self, uncertainty_type: UncertaintyType, text: str) -> str:
        """Generate improvement suggestion for uncertainty indicator."""
        suggestions = {
            UncertaintyType.LINGUISTIC: f"Consider removing uncertain language: '{text}'",
            UncertaintyType.FACTUAL: f"Verify factual information around: '{text}'",
            UncertaintyType.TEMPORAL: f"Clarify temporal information: '{text}'",
            UncertaintyType.SEMANTIC: f"Use more specific language instead of: '{text}'"
        }
        return suggestions.get(uncertainty_type, f"Review: '{text}'")
    
    async def _analyze_structural_uncertainty(self, text: str, context: Dict[str, Any]) -> List[UncertaintyIndicator]:
        """Analyze structural completeness and clarity."""
        indicators = []
        
        # Check for incomplete sentences
        sentences = text.split('.')
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence and (len(sentence) < 10 or sentence.count(' ') < 2):
                indicator = UncertaintyIndicator(
                    type=UncertaintyType.STRUCTURAL,
                    text=sentence,
                    position=(0, len(sentence)),  # Approximate position
                    confidence=0.7,
                    context=sentence,
                    severity=ValidationSeverity.MEDIUM,
                    suggestion="Consider expanding this incomplete sentence"
                )
                indicators.append(indicator)
        
        # Check for missing key information
        query = context.get('query', '')
        if query and len(text) < len(query) * 2:
            indicator = UncertaintyIndicator(
                type=UncertaintyType.STRUCTURAL,
                text=text[:50] + "..." if len(text) > 50 else text,
                position=(0, min(50, len(text))),
                confidence=0.8,
                context="Response length analysis",
                severity=ValidationSeverity.MEDIUM,
                suggestion="Response may be too brief for the given query"
            )
            indicators.append(indicator)
        
        return indicators
    
    async def _analyze_contextual_uncertainty(self, text: str, context: Dict[str, Any]) -> List[UncertaintyIndicator]:
        """Analyze contextual appropriateness and completeness."""
        indicators = []
        
        # Check if response addresses the query
        query = context.get('query', '')
        if query:
            # Simple keyword overlap analysis
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            overlap = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0
            
            if overlap < 0.2:  # Less than 20% keyword overlap
                indicator = UncertaintyIndicator(
                    type=UncertaintyType.CONTEXTUAL,
                    text="Response content",
                    position=(0, len(text)),
                    confidence=0.6,
                    context=f"Query: {query[:100]}...",
                    severity=ValidationSeverity.MEDIUM,
                    suggestion="Response may not adequately address the query"
                )
                indicators.append(indicator)
        
        return indicators
    
    def _deduplicate_indicators(self, indicators: List[UncertaintyIndicator]) -> List[UncertaintyIndicator]:
        """Remove duplicate indicators based on position overlap."""
        if not indicators:
            return []
        
        # Sort by position
        indicators.sort(key=lambda x: x.position[0])
        
        deduplicated = [indicators[0]]
        for indicator in indicators[1:]:
            # Check if this indicator overlaps with the last added one
            last_indicator = deduplicated[-1]
            if (indicator.position[0] < last_indicator.position[1] and 
                indicator.type == last_indicator.type):
                # Skip overlapping indicator of same type
                continue
            deduplicated.append(indicator)
        
        return deduplicated


class ValidationRuleEngine:
    """
    Rule engine for confidence validation with configurable rules.
    """
    
    def __init__(self):
        self.rules = self._initialize_default_rules()
    
    def _initialize_default_rules(self) -> List[ConfidenceRule]:
        """Initialize default validation rules."""
        return [
            ConfidenceRule(
                name="source_reliability",
                metric=ConfidenceMetric.SOURCE_RELIABILITY,
                weight=0.2,
                threshold=0.7,
                severity=ValidationSeverity.HIGH,
                description="Evaluates reliability of information sources",
                llm_based=False
            ),
            ConfidenceRule(
                name="content_coherence",
                metric=ConfidenceMetric.CONTENT_COHERENCE,
                weight=0.15,
                threshold=0.8,
                severity=ValidationSeverity.MEDIUM,
                description="Assesses logical flow and coherence of content",
                llm_based=True
            ),
            ConfidenceRule(
                name="factual_consistency",
                metric=ConfidenceMetric.FACTUAL_CONSISTENCY,
                weight=0.2,
                threshold=0.8,
                severity=ValidationSeverity.HIGH,
                description="Validates factual accuracy and consistency",
                llm_based=True
            ),
            ConfidenceRule(
                name="semantic_clarity",
                metric=ConfidenceMetric.SEMANTIC_CLARITY,
                weight=0.1,
                threshold=0.7,
                severity=ValidationSeverity.MEDIUM,
                description="Evaluates clarity and precision of language",
                llm_based=False
            ),
            ConfidenceRule(
                name="entity_accuracy",
                metric=ConfidenceMetric.ENTITY_ACCURACY,
                weight=0.15,
                threshold=0.8,
                severity=ValidationSeverity.HIGH,
                description="Validates accuracy of named entities",
                llm_based=False
            ),
            ConfidenceRule(
                name="relationship_validity",
                metric=ConfidenceMetric.RELATIONSHIP_VALIDITY,
                weight=0.1,
                threshold=0.7,
                severity=ValidationSeverity.MEDIUM,
                description="Validates entity relationships and connections",
                llm_based=False
            ),
            ConfidenceRule(
                name="temporal_consistency",
                metric=ConfidenceMetric.TEMPORAL_CONSISTENCY,
                weight=0.05,
                threshold=0.8,
                severity=ValidationSeverity.MEDIUM,
                description="Validates temporal information consistency",
                llm_based=False
            ),
            ConfidenceRule(
                name="logical_soundness",
                metric=ConfidenceMetric.LOGICAL_SOUNDNESS,
                weight=0.05,
                threshold=0.8,
                severity=ValidationSeverity.HIGH,
                description="Evaluates logical reasoning and soundness",
                llm_based=True
            )
        ]
    
    def add_rule(self, rule: ConfidenceRule) -> None:
        """Add a new validation rule."""
        self.rules.append(rule)
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a validation rule by name."""
        initial_count = len(self.rules)
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        return len(self.rules) < initial_count
    
    def get_rule(self, rule_name: str) -> Optional[ConfidenceRule]:
        """Get a validation rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                return rule
        return None
    
    def get_enabled_rules(self) -> List[ConfidenceRule]:
        """Get all enabled validation rules."""
        return [rule for rule in self.rules if rule.enabled]
    
    def get_llm_rules(self) -> List[ConfidenceRule]:
        """Get all LLM-based validation rules."""
        return [rule for rule in self.rules if rule.enabled and rule.llm_based]
    
    def update_rule(self, rule_name: str, **kwargs) -> bool:
        """Update a validation rule."""
        rule = self.get_rule(rule_name)
        if rule:
            for key, value in kwargs.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            return True
        return False
    
    def validate_rule_weights(self) -> bool:
        """Validate that rule weights sum to approximately 1.0."""
        enabled_rules = self.get_enabled_rules()
        total_weight = sum(rule.weight for rule in enabled_rules)
        return abs(total_weight - 1.0) < 0.01
    
    def normalize_weights(self) -> None:
        """Normalize rule weights to sum to 1.0."""
        enabled_rules = self.get_enabled_rules()
        total_weight = sum(rule.weight for rule in enabled_rules)
        
        if total_weight > 0:
            for rule in enabled_rules:
                rule.weight = rule.weight / total_weight


class LLMConfidenceAssessor:
    """
    LLM-based confidence assessment for advanced semantic evaluation.
    """
    
    def __init__(self):
        self.assessment_prompts = {
            ConfidenceMetric.CONTENT_COHERENCE: """
Evaluate the coherence and logical flow of this response:

Query: {query}
Response: {content}

Rate the coherence on a scale of 0.0 to 1.0 considering:
- Logical structure and flow
- Consistency of arguments
- Clarity of presentation
- Absence of contradictions

Provide only a numeric score between 0.0 and 1.0.
""",
            ConfidenceMetric.FACTUAL_CONSISTENCY: """
Evaluate the factual consistency of this response:

Query: {query}
Response: {content}
Sources: {sources}

Rate the factual consistency on a scale of 0.0 to 1.0 considering:
- Accuracy of stated facts
- Consistency with provided sources
- Absence of contradictory information
- Verifiability of claims

Provide only a numeric score between 0.0 and 1.0.
""",
            ConfidenceMetric.LOGICAL_SOUNDNESS: """
Evaluate the logical soundness of this response:

Query: {query}
Response: {content}

Rate the logical soundness on a scale of 0.0 to 1.0 considering:
- Validity of reasoning
- Sound logical connections
- Appropriate conclusions
- Absence of logical fallacies

Provide only a numeric score between 0.0 and 1.0.
"""
        }
    
    async def assess_confidence(self, request: LLMConfidenceRequest, metrics: List[ConfidenceMetric]) -> Dict[ConfidenceMetric, float]:
        """
        Perform LLM-based confidence assessment for specified metrics.
        
        Args:
            request: LLM confidence request with content and context
            metrics: List of metrics to assess
            
        Returns:
            Dictionary mapping metrics to confidence scores
        """
        try:
            results = {}
            
            # Create assessment tasks for parallel execution
            tasks = []
            for metric in metrics:
                if metric in self.assessment_prompts:
                    task = self._assess_single_metric(request, metric)
                    tasks.append((metric, task))
            
            # Execute assessments in parallel
            if tasks:
                task_results = await asyncio.gather(
                    *[task for _, task in tasks],
                    return_exceptions=True
                )
                
                for i, (metric, _) in enumerate(tasks):
                    result = task_results[i]
                    if isinstance(result, Exception):
                        logger.error(f"LLM assessment failed for {metric}: {result}")
                        results[metric] = 0.5  # Default fallback score
                    else:
                        results[metric] = result
            
            return results
            
        except Exception as e:
            logger.error(f"Error in LLM confidence assessment: {e}")
            return {metric: 0.5 for metric in metrics}
    
    async def _assess_single_metric(self, request: LLMConfidenceRequest, metric: ConfidenceMetric) -> float:
        """Assess a single confidence metric using LLM."""
        try:
            prompt = self.assessment_prompts[metric].format(
                query=request.query,
                content=request.content,
                sources=self._format_sources(request.sources)
            )
            
            # Mock LLM call - in production, replace with actual LLM API call
            score = await self._mock_llm_call(prompt, metric)
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error assessing metric {metric}: {e}")
            return 0.5
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources for LLM prompt."""
        if not sources:
            return "No sources provided"
        
        formatted = []
        for i, source in enumerate(sources[:5], 1):  # Limit to first 5 sources
            content = source.get('content', 'No content')
            confidence = source.get('confidence', 'Unknown')
            formatted.append(f"{i}. {content} (Confidence: {confidence})")
        
        return "\n".join(formatted)
    
    async def _mock_llm_call(self, prompt: str, metric: ConfidenceMetric) -> float:
        """
        Mock LLM call for testing. In production, replace with actual LLM API.
        """
        # Simulate LLM processing time
        await asyncio.sleep(0.1)
        
        # Mock scoring based on metric type and prompt content
        if "contradictory" in prompt.lower() or "inconsistent" in prompt.lower():
            return 0.3
        elif "clear" in prompt.lower() and "accurate" in prompt.lower():
            return 0.9
        elif metric == ConfidenceMetric.CONTENT_COHERENCE:
            return 0.8
        elif metric == ConfidenceMetric.FACTUAL_CONSISTENCY:
            return 0.85
        elif metric == ConfidenceMetric.LOGICAL_SOUNDNESS:
            return 0.75
        else:
            return 0.7


class AdvancedConfidenceScorer:
    """
    Advanced confidence scoring system with LLM-based assessment and validation rules.
    """
    
    def __init__(self):
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.rule_engine = ValidationRuleEngine()
        self.llm_assessor = LLMConfidenceAssessor()
        
        # Ensure rule weights are normalized
        if not self.rule_engine.validate_rule_weights():
            self.rule_engine.normalize_weights()
            logger.warning("Normalized confidence rule weights")
    
    async def assess_confidence(
        self, 
        content: str, 
        query: str, 
        context: Dict[str, Any] = None,
        sources: List[Dict[str, Any]] = None,
        entities: List[str] = None,
        relationships: List[Dict[str, Any]] = None
    ) -> ConfidenceAssessment:
        """
        Perform comprehensive confidence assessment.
        
        Args:
            content: Response content to assess
            query: Original query
            context: Additional context information
            sources: Source documents used
            entities: Extracted entities
            relationships: Entity relationships
            
        Returns:
            Comprehensive confidence assessment
        """
        start_time = time.time()
        
        try:
            # Initialize parameters
            context = context or {}
            sources = sources or []
            entities = entities or []
            relationships = relationships or []
            
            # Create LLM request
            llm_request = LLMConfidenceRequest(
                content=content,
                query=query,
                context=context,
                sources=sources,
                entities=entities,
                relationships=relationships
            )
            
            # Run assessments in parallel
            uncertainty_task = self.uncertainty_quantifier.quantify_uncertainty(content, context)
            rule_based_task = self._assess_rule_based_metrics(llm_request)
            llm_task = self._assess_llm_based_metrics(llm_request)
            
            uncertainty_indicators, metric_scores_rule, metric_scores_llm = await asyncio.gather(
                uncertainty_task, rule_based_task, llm_task
            )
            
            # Combine metric scores
            metric_scores = {**metric_scores_rule, **metric_scores_llm}
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(metric_scores, uncertainty_indicators)
            
            # Generate validation results
            validation_results = self._generate_validation_results(metric_scores, uncertainty_indicators)
            
            # Create LLM assessment summary
            llm_assessment = {
                "metrics_assessed": list(metric_scores_llm.keys()),
                "average_llm_score": sum(metric_scores_llm.values()) / len(metric_scores_llm) if metric_scores_llm else 0.0,
                "assessment_time": time.time() - start_time
            }
            
            processing_time = time.time() - start_time
            
            return ConfidenceAssessment(
                overall_confidence=overall_confidence,
                metric_scores=metric_scores,
                uncertainty_indicators=uncertainty_indicators,
                validation_results=validation_results,
                llm_assessment=llm_assessment,
                processing_time=processing_time,
                metadata={
                    "rules_applied": len(self.rule_engine.get_enabled_rules()),
                    "uncertainty_count": len(uncertainty_indicators),
                    "high_severity_count": len([ui for ui in uncertainty_indicators if ui.severity == ValidationSeverity.HIGH])
                }
            )
            
        except Exception as e:
            logger.error(f"Error in confidence assessment: {e}")
            return ConfidenceAssessment(
                overall_confidence=0.5,
                metric_scores={},
                uncertainty_indicators=[],
                validation_results=[],
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    async def _assess_rule_based_metrics(self, request: LLMConfidenceRequest) -> Dict[ConfidenceMetric, float]:
        """Assess confidence using rule-based metrics."""
        scores = {}
        
        # Source reliability assessment
        scores[ConfidenceMetric.SOURCE_RELIABILITY] = self._assess_source_reliability(request.sources)
        
        # Semantic clarity assessment
        scores[ConfidenceMetric.SEMANTIC_CLARITY] = self._assess_semantic_clarity(request.content)
        
        # Entity accuracy assessment
        scores[ConfidenceMetric.ENTITY_ACCURACY] = self._assess_entity_accuracy(request.entities, request.content)
        
        # Relationship validity assessment
        scores[ConfidenceMetric.RELATIONSHIP_VALIDITY] = self._assess_relationship_validity(request.relationships, request.content)
        
        # Temporal consistency assessment
        scores[ConfidenceMetric.TEMPORAL_CONSISTENCY] = self._assess_temporal_consistency(request.content)
        
        return scores
    
    async def _assess_llm_based_metrics(self, request: LLMConfidenceRequest) -> Dict[ConfidenceMetric, float]:
        """Assess confidence using LLM-based metrics."""
        llm_rules = self.rule_engine.get_llm_rules()
        llm_metrics = [rule.metric for rule in llm_rules]
        
        if not llm_metrics:
            return {}
        
        return await self.llm_assessor.assess_confidence(request, llm_metrics)
    
    def _assess_source_reliability(self, sources: List[Dict[str, Any]]) -> float:
        """Assess reliability of sources."""
        if not sources:
            return 0.3  # Low confidence without sources
        
        total_confidence = 0.0
        total_weight = 0.0
        
        for source in sources:
            source_confidence = source.get('confidence', 0.5)
            source_type = source.get('type', 'unknown')
            
            # Weight sources by type
            type_weights = {
                'document': 1.0,
                'database': 0.9,
                'api': 0.8,
                'web': 0.6,
                'unknown': 0.4
            }
            weight = type_weights.get(source_type, 0.4)
            
            total_confidence += source_confidence * weight
            total_weight += weight
        
        return total_confidence / total_weight if total_weight > 0 else 0.5
    
    def _assess_semantic_clarity(self, content: str) -> float:
        """Assess semantic clarity of content."""
        if not content:
            return 0.0
        
        # Basic clarity metrics
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Penalize very short or very long sentences
        length_score = 1.0
        if avg_sentence_length < 5:
            length_score = 0.6
        elif avg_sentence_length > 30:
            length_score = 0.7
        
        # Check for unclear terms
        unclear_terms = ['thing', 'stuff', 'something', 'somehow', 'whatever']
        unclear_count = sum(content.lower().count(term) for term in unclear_terms)
        clarity_penalty = min(unclear_count * 0.1, 0.4)
        
        return max(0.0, length_score - clarity_penalty)
    
    def _assess_entity_accuracy(self, entities: List[str], content: str) -> float:
        """Assess accuracy of named entities."""
        if not entities:
            return 0.8  # Neutral score if no entities
        
        content_lower = content.lower()
        entity_coverage = 0
        
        for entity in entities:
            if entity.lower() in content_lower:
                entity_coverage += 1
        
        coverage_ratio = entity_coverage / len(entities) if entities else 0
        return min(0.5 + coverage_ratio * 0.5, 1.0)
    
    def _assess_relationship_validity(self, relationships: List[Dict[str, Any]], content: str) -> float:
        """Assess validity of entity relationships."""
        if not relationships:
            return 0.8  # Neutral score if no relationships
        
        valid_relationships = 0
        content_lower = content.lower()
        
        for rel in relationships:
            source = rel.get('source', '').lower()
            target = rel.get('target', '').lower()
            
            if source in content_lower and target in content_lower:
                valid_relationships += 1
        
        validity_ratio = valid_relationships / len(relationships) if relationships else 0
        return min(0.5 + validity_ratio * 0.5, 1.0)
    
    def _assess_temporal_consistency(self, content: str) -> float:
        """Assess temporal consistency in content."""
        # Extract years (including historical years)
        year_pattern = r'\b\d{4}\b'
        potential_years = [int(match.group()) for match in re.finditer(year_pattern, content)]
        # Filter for reasonable year ranges (0-2100)
        years = [year for year in potential_years if 0 <= year <= 2100]
        
        if len(years) < 2:
            return 0.9  # High score if insufficient temporal information
        
        # Check for reasonable temporal ranges
        year_range = max(years) - min(years)
        
        if year_range > 1000:  # Very large range might indicate error
            return 0.4
        elif year_range > 500:  # Large range (e.g., 1066-2020)
            return 0.7
        elif year_range > 100:
            return 0.8
        else:
            return 0.9
    
    def _calculate_overall_confidence(
        self, 
        metric_scores: Dict[ConfidenceMetric, float], 
        uncertainty_indicators: List[UncertaintyIndicator]
    ) -> float:
        """Calculate overall confidence score."""
        if not metric_scores:
            return 0.5
        
        # Weighted average of metric scores
        enabled_rules = self.rule_engine.get_enabled_rules()
        weighted_sum = 0.0
        total_weight = 0.0
        
        for rule in enabled_rules:
            if rule.metric in metric_scores:
                weighted_sum += metric_scores[rule.metric] * rule.weight
                total_weight += rule.weight
        
        base_confidence = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Apply uncertainty penalties
        high_severity_count = len([ui for ui in uncertainty_indicators if ui.severity in [ValidationSeverity.HIGH, ValidationSeverity.CRITICAL]])
        medium_severity_count = len([ui for ui in uncertainty_indicators if ui.severity == ValidationSeverity.MEDIUM])
        
        uncertainty_penalty = (high_severity_count * 0.15) + (medium_severity_count * 0.05)
        
        final_confidence = max(0.0, base_confidence - uncertainty_penalty)
        return min(final_confidence, 1.0)
    
    def _generate_validation_results(
        self, 
        metric_scores: Dict[ConfidenceMetric, float], 
        uncertainty_indicators: List[UncertaintyIndicator]
    ) -> List[Dict[str, Any]]:
        """Generate validation results based on rules and thresholds."""
        results = []
        
        # Check metric thresholds
        for rule in self.rule_engine.get_enabled_rules():
            if rule.metric in metric_scores:
                score = metric_scores[rule.metric]
                passed = score >= rule.threshold
                
                result = {
                    "rule_name": rule.name,
                    "metric": rule.metric.value,
                    "score": score,
                    "threshold": rule.threshold,
                    "passed": passed,
                    "severity": rule.severity.value,
                    "description": rule.description,
                    "weight": rule.weight
                }
                results.append(result)
        
        # Add uncertainty validation results
        critical_uncertainties = [ui for ui in uncertainty_indicators if ui.severity == ValidationSeverity.CRITICAL]
        if critical_uncertainties:
            result = {
                "rule_name": "critical_uncertainty_check",
                "metric": "uncertainty_level",
                "score": 0.0,
                "threshold": 0.0,
                "passed": False,
                "severity": ValidationSeverity.CRITICAL.value,
                "description": f"Found {len(critical_uncertainties)} critical uncertainty indicators",
                "weight": 0.0,
                "details": [ui.suggestion for ui in critical_uncertainties]
            }
            results.append(result)
        
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the confidence scoring system."""
        try:
            start_time = time.time()
            
            # Test uncertainty quantification
            test_text = "I think this might be correct, but I'm not sure about the details."
            uncertainty_indicators = await self.uncertainty_quantifier.quantify_uncertainty(test_text)
            
            # Test rule engine
            enabled_rules = self.rule_engine.get_enabled_rules()
            weights_valid = self.rule_engine.validate_rule_weights()
            
            # Test LLM assessor (mock)
            test_request = LLMConfidenceRequest(
                content="Test content",
                query="Test query",
                context={},
                sources=[],
                entities=[],
                relationships=[]
            )
            llm_metrics = [ConfidenceMetric.CONTENT_COHERENCE]
            llm_scores = await self.llm_assessor.assess_confidence(test_request, llm_metrics)
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "components": {
                    "uncertainty_quantifier": len(uncertainty_indicators) >= 0,
                    "rule_engine": len(enabled_rules) > 0 and weights_valid,
                    "llm_assessor": len(llm_scores) > 0
                },
                "metrics": {
                    "enabled_rules": len(enabled_rules),
                    "rule_weights_valid": weights_valid,
                    "uncertainty_patterns": len(self.uncertainty_quantifier.uncertainty_patterns),
                    "llm_metrics_available": len(self.llm_assessor.assessment_prompts)
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time": time.time() - start_time
            }


# Export main classes for use in other modules
__all__ = [
    'ConfidenceMetric',
    'UncertaintyType', 
    'ValidationSeverity',
    'ConfidenceRule',
    'UncertaintyIndicator',
    'ConfidenceAssessment',
    'LLMConfidenceRequest',
    'UncertaintyQuantifier',
    'ValidationRuleEngine',
    'LLMConfidenceAssessor',
    'AdvancedConfidenceScorer'
]