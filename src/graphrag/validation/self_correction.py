"""
Multi-Layer Self-Correction Framework for Phase 2 Hybrid RAG System

Implements comprehensive validation, confidence scoring, consistency checking,
and automated correction mechanisms to reduce hallucinations and improve accuracy.
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from graphrag.config.settings import Config
from graphrag.logger import logger


class ValidationType(Enum):
    """Types of validation checks."""
    
    ENTITY_CONSISTENCY = "entity_consistency"
    RELATIONSHIP_CONSISTENCY = "relationship_consistency"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    FACTUAL_CONSISTENCY = "factual_consistency"
    SOURCE_VERIFICATION = "source_verification"
    HALLUCINATION_DETECTION = "hallucination_detection"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CorrectionAction(Enum):
    """Types of correction actions."""
    
    ACCEPT = "accept"
    MODIFY = "modify"
    REGENERATE = "regenerate"
    REJECT = "reject"
    FLAG_UNCERTAIN = "flag_uncertain"


@dataclass
class ValidationRule:
    """Represents a validation rule with configuration."""
    
    rule_type: ValidationType
    enabled: bool
    weight: float
    threshold: float
    description: str
    error_message: str = ""


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    rule_type: ValidationType
    passed: bool
    confidence: float
    severity: ValidationSeverity
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)
    execution_time: float = 0.0


@dataclass
class ConfidenceScore:
    """Comprehensive confidence scoring."""
    
    source_confidence: float
    consistency_confidence: float
    completeness_confidence: float
    overall_confidence: float
    confidence_factors: dict[str, float] = field(default_factory=dict)
    uncertainty_flags: list[str] = field(default_factory=list)


@dataclass
class CorrectionSuggestion:
    """Suggested correction for validation issues."""
    
    action: CorrectionAction
    confidence: float
    original_text: str
    corrected_text: str
    reasoning: str
    validation_results: list[ValidationResult] = field(default_factory=list)


@dataclass
class RAGResponse:
    """Enhanced RAG response with validation metadata."""
    
    content: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    confidence_score: Optional[ConfidenceScore] = None
    validation_results: list[ValidationResult] = field(default_factory=list)
    correction_suggestions: list[CorrectionSuggestion] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class HallucinationDetector:
    """Detects potential hallucinations in generated responses."""
    
    def __init__(self):
        self.hallucination_patterns = [
            r"(I think|I believe|I assume|It seems|probably|maybe|possibly)",
            r"(as far as I know|to my knowledge|I'm not certain)",
            r"(could be|might be|may be|it's possible)",
            r"(according to my understanding|in my opinion)"
        ]
        self.factual_inconsistency_patterns = [
            r"(\d{4}) and (\d{4})",  # Conflicting years
            r"(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}), (\d{4})",  # Date patterns
        ]
    
    async def detect_uncertainty_language(self, text: str) -> dict[str, Any]:
        """
        Detect uncertainty language that may indicate hallucinations.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with detection results
        """
        uncertainty_matches = []
        uncertainty_score = 0.0
        
        for pattern in self.hallucination_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                uncertainty_matches.append({
                    "pattern": pattern,
                    "match": match.group(),
                    "position": match.span(),
                    "context": text[max(0, match.start()-20):match.end()+20]
                })
                uncertainty_score += 0.1
        
        return {
            "uncertainty_score": min(uncertainty_score, 1.0),
            "matches": uncertainty_matches,
            "has_uncertainty": len(uncertainty_matches) > 0
        }
    
    async def detect_factual_inconsistencies(self, text: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Detect potential factual inconsistencies.
        
        Args:
            text: Text to analyze
            context: Context information for verification
            
        Returns:
            Dictionary with inconsistency detection results
        """
        inconsistencies = []
        
        # Check for conflicting dates
        date_matches = re.finditer(r"(\d{4})", text)
        years = [int(match.group()) for match in date_matches]
        
        if len(set(years)) > 1:
            year_range = max(years) - min(years)
            if year_range > 50:  # Large year gaps might indicate inconsistency
                inconsistencies.append({
                    "type": "temporal_inconsistency",
                    "description": f"Large year range detected: {min(years)}-{max(years)}",
                    "severity": "medium"
                })
        
        # Check for impossible statements
        impossible_patterns = [
            r"(was born after.*died)",
            r"(founded before.*was born)",
            r"(happened in \d{4}.*before \d{4})"  # Event order inconsistencies
        ]
        
        for pattern in impossible_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                inconsistencies.append({
                    "type": "logical_inconsistency",
                    "description": f"Possible logical inconsistency detected",
                    "pattern": pattern,
                    "severity": "high"
                })
        
        return {
            "inconsistency_score": min(len(inconsistencies) * 0.2, 1.0),
            "inconsistencies": inconsistencies,
            "has_inconsistencies": len(inconsistencies) > 0
        }


class ConfidenceScorer:
    """Calculates comprehensive confidence scores for responses."""
    
    def __init__(self):
        self.source_weight = Config.CONFIDENCE_SOURCE_WEIGHT
        self.consistency_weight = Config.CONFIDENCE_CONSISTENCY_WEIGHT
        self.completeness_weight = Config.CONFIDENCE_COMPLETENESS_WEIGHT
    
    async def calculate_source_confidence(self, response: RAGResponse) -> float:
        """
        Calculate confidence based on source quality and reliability.
        
        Args:
            response: RAG response with sources
            
        Returns:
            Source confidence score (0.0 to 1.0)
        """
        if not response.sources:
            return 0.3  # Low confidence without sources
        
        source_scores = []
        
        for source in response.sources:
            score = 0.5  # Base score
            
            # Check source metadata
            if source.get("confidence", 0) > 0.8:
                score += 0.2
            
            if source.get("type") == "document":
                score += 0.1
            elif source.get("type") == "database":
                score += 0.2
            
            # Check source recency
            if source.get("timestamp"):
                try:
                    source_time = datetime.fromisoformat(str(source["timestamp"]))
                    days_old = (datetime.now() - source_time).days
                    if days_old < 30:
                        score += 0.1
                    elif days_old > 365:
                        score -= 0.1
                except:
                    pass
            
            source_scores.append(min(score, 1.0))
        
        return sum(source_scores) / len(source_scores)
    
    async def calculate_consistency_confidence(self, response: RAGResponse, context: dict[str, Any]) -> float:
        """
        Calculate confidence based on internal consistency.
        
        Args:
            response: RAG response to analyze
            context: Additional context for consistency checking
            
        Returns:
            Consistency confidence score (0.0 to 1.0)
        """
        consistency_score = 0.8  # Start with high consistency assumption
        
        # Check entity consistency
        entities = response.entities
        if len(entities) > 0:
            # Check for entity name variations
            entity_variations = {}
            for entity in entities:
                base_name = entity.lower().strip()
                if base_name in entity_variations:
                    consistency_score -= 0.1
                entity_variations[base_name] = entity
        
        # Check relationship consistency
        if response.relationships:
            relationship_pairs = set()
            for rel in response.relationships:
                source = rel.get("source", "").lower()
                target = rel.get("target", "").lower()
                rel_type = rel.get("type", "").lower()
                
                # Check for conflicting relationships
                reverse_pair = (target, source, rel_type)
                if reverse_pair in relationship_pairs:
                    consistency_score -= 0.15
                
                relationship_pairs.add((source, target, rel_type))
        
        # Check content consistency with sources
        if response.sources and response.content:
            content_lower = response.content.lower()
            source_alignment = 0
            
            for source in response.sources:
                source_content = str(source.get("content", "")).lower()
                if source_content and len(source_content) > 10:
                    # Simple overlap check
                    common_words = set(content_lower.split()) & set(source_content.split())
                    if len(common_words) > 3:
                        source_alignment += 1
            
            if response.sources and source_alignment == 0:
                consistency_score -= 0.2
        
        return max(consistency_score, 0.0)
    
    async def calculate_completeness_confidence(self, response: RAGResponse, query: str) -> float:
        """
        Calculate confidence based on response completeness.
        
        Args:
            response: RAG response to analyze
            query: Original query
            
        Returns:
            Completeness confidence score (0.0 to 1.0)
        """
        completeness_score = 0.5  # Base score
        
        # Check response length
        if len(response.content) > 50:
            completeness_score += 0.2
        if len(response.content) > 200:
            completeness_score += 0.1
        
        # Check if response addresses query keywords
        query_words = set(query.lower().split())
        response_words = set(response.content.lower().split())
        
        keyword_overlap = len(query_words & response_words) / max(len(query_words), 1)
        completeness_score += keyword_overlap * 0.3
        
        # Check for specific elements
        if response.entities:
            completeness_score += 0.1
        
        if response.relationships:
            completeness_score += 0.1
        
        if response.sources:
            completeness_score += 0.1
        
        return min(completeness_score, 1.0)
    
    async def calculate_confidence_score(self, response: RAGResponse, query: str, context: dict[str, Any]) -> ConfidenceScore:
        """
        Calculate comprehensive confidence score.
        
        Args:
            response: RAG response to analyze
            query: Original query
            context: Additional context
            
        Returns:
            Complete confidence score breakdown
        """
        # Calculate individual confidence components
        source_conf = await self.calculate_source_confidence(response)
        consistency_conf = await self.calculate_consistency_confidence(response, context)
        completeness_conf = await self.calculate_completeness_confidence(response, query)
        
        # Calculate weighted overall confidence
        overall_conf = (
            source_conf * self.source_weight +
            consistency_conf * self.consistency_weight +
            completeness_conf * self.completeness_weight
        )
        
        # Identify uncertainty flags
        uncertainty_flags = []
        if source_conf < 0.5:
            uncertainty_flags.append("Low source confidence")
        if consistency_conf < 0.6:
            uncertainty_flags.append("Consistency issues detected")
        if completeness_conf < 0.5:
            uncertainty_flags.append("Incomplete response")
        
        return ConfidenceScore(
            source_confidence=source_conf,
            consistency_confidence=consistency_conf,
            completeness_confidence=completeness_conf,
            overall_confidence=overall_conf,
            confidence_factors={
                "source_weight": self.source_weight,
                "consistency_weight": self.consistency_weight,
                "completeness_weight": self.completeness_weight
            },
            uncertainty_flags=uncertainty_flags
        )


class ValidationEngine:
    """Multi-layer validation engine for RAG responses."""
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
        self.hallucination_detector = HallucinationDetector()
        self.confidence_scorer = ConfidenceScorer()
    
    def _initialize_validation_rules(self) -> dict[ValidationType, ValidationRule]:
        """Initialize validation rules from configuration."""
        return {
            ValidationType.ENTITY_CONSISTENCY: ValidationRule(
                rule_type=ValidationType.ENTITY_CONSISTENCY,
                enabled=Config.VALIDATION_ENTITY_CONSISTENCY,
                weight=0.25,
                threshold=0.7,
                description="Check entity name consistency and disambiguation",
                error_message="Entity consistency issues detected"
            ),
            ValidationType.RELATIONSHIP_CONSISTENCY: ValidationRule(
                rule_type=ValidationType.RELATIONSHIP_CONSISTENCY,
                enabled=Config.VALIDATION_RELATIONSHIP_CONSISTENCY,
                weight=0.25,
                threshold=0.6,
                description="Validate relationship logic and consistency",
                error_message="Relationship consistency issues detected"
            ),
            ValidationType.TEMPORAL_CONSISTENCY: ValidationRule(
                rule_type=ValidationType.TEMPORAL_CONSISTENCY,
                enabled=Config.VALIDATION_TEMPORAL_CONSISTENCY,
                weight=0.2,
                threshold=0.8,
                description="Check temporal order and date consistency",
                error_message="Temporal inconsistencies detected"
            ),
            ValidationType.FACTUAL_CONSISTENCY: ValidationRule(
                rule_type=ValidationType.FACTUAL_CONSISTENCY,
                enabled=Config.VALIDATION_FACTUAL_CONSISTENCY,
                weight=0.3,
                threshold=0.7,
                description="Verify factual accuracy and logical consistency",
                error_message="Factual inconsistencies detected"
            ),
            ValidationType.HALLUCINATION_DETECTION: ValidationRule(
                rule_type=ValidationType.HALLUCINATION_DETECTION,
                enabled=Config.HALLUCINATION_DETECTION_ENABLED,
                weight=0.3,
                threshold=Config.HALLUCINATION_THRESHOLD,
                description="Detect potential hallucinations and uncertain statements",
                error_message="Potential hallucinations detected"
            )
        }
    
    async def validate_entity_consistency(self, response: RAGResponse, context: dict[str, Any]) -> ValidationResult:
        """Validate entity consistency within the response."""
        start_time = time.time()
        
        try:
            if not response.entities:
                return ValidationResult(
                    rule_type=ValidationType.ENTITY_CONSISTENCY,
                    passed=True,
                    confidence=1.0,
                    severity=ValidationSeverity.LOW,
                    message="No entities to validate",
                    execution_time=time.time() - start_time
                )
            
            # Check for entity name variations
            entity_groups = {}
            issues = []
            
            for entity in response.entities:
                normalized = entity.lower().strip()
                
                # Group similar entities
                found_group = False
                for group_key in entity_groups:
                    if self._entities_similar(normalized, group_key):
                        entity_groups[group_key].append(entity)
                        found_group = True
                        break
                
                if not found_group:
                    entity_groups[normalized] = [entity]
            
            # Check for inconsistencies
            for group_entities in entity_groups.values():
                if len(group_entities) > 1:
                    issues.append(f"Entity variations found: {', '.join(group_entities)}")
            
            consistency_score = max(0.0, 1.0 - (len(issues) * 0.2))
            passed = consistency_score >= self.validation_rules[ValidationType.ENTITY_CONSISTENCY].threshold
            
            return ValidationResult(
                rule_type=ValidationType.ENTITY_CONSISTENCY,
                passed=passed,
                confidence=consistency_score,
                severity=ValidationSeverity.MEDIUM if not passed else ValidationSeverity.LOW,
                message=f"Entity consistency check completed. Issues: {len(issues)}",
                evidence={"issues": issues, "entity_groups": len(entity_groups)},
                suggestions=[f"Standardize entity names: {issue}" for issue in issues],
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Entity consistency validation failed: {e}")
            return ValidationResult(
                rule_type=ValidationType.ENTITY_CONSISTENCY,
                passed=False,
                confidence=0.0,
                severity=ValidationSeverity.HIGH,
                message=f"Validation error: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _entities_similar(self, entity1: str, entity2: str) -> bool:
        """Check if two entity names are similar enough to be the same entity."""
        # Simple similarity check
        if entity1 == entity2:
            return True
        
        # Check if one is a substring of the other
        if entity1 in entity2 or entity2 in entity1:
            return True
        
        # Check common variations
        entity1_words = set(entity1.split())
        entity2_words = set(entity2.split())
        
        # If they share most words, consider them similar
        common_words = entity1_words & entity2_words
        total_words = entity1_words | entity2_words
        
        similarity = len(common_words) / len(total_words) if total_words else 0
        return similarity > 0.7
    
    async def validate_relationship_consistency(self, response: RAGResponse, context: dict[str, Any]) -> ValidationResult:
        """Validate relationship logic and consistency."""
        start_time = time.time()
        
        try:
            if not response.relationships:
                return ValidationResult(
                    rule_type=ValidationType.RELATIONSHIP_CONSISTENCY,
                    passed=True,
                    confidence=1.0,
                    severity=ValidationSeverity.LOW,
                    message="No relationships to validate",
                    execution_time=time.time() - start_time
                )
            
            issues = []
            relationship_map = {}
            
            # Build relationship map and check for conflicts
            for rel in response.relationships:
                source = rel.get("source", "").lower()
                target = rel.get("target", "").lower()
                rel_type = rel.get("type", "").lower()
                
                if not source or not target or not rel_type:
                    issues.append("Incomplete relationship found")
                    continue
                
                key = (source, target)
                if key in relationship_map:
                    if relationship_map[key] != rel_type:
                        issues.append(f"Conflicting relationships: {source} -> {target}")
                else:
                    relationship_map[key] = rel_type
                
                # Check for impossible relationships
                if self._is_impossible_relationship(source, target, rel_type, context):
                    issues.append(f"Potentially impossible relationship: {source} {rel_type} {target}")
            
            consistency_score = max(0.0, 1.0 - (len(issues) * 0.15))
            passed = consistency_score >= self.validation_rules[ValidationType.RELATIONSHIP_CONSISTENCY].threshold
            
            return ValidationResult(
                rule_type=ValidationType.RELATIONSHIP_CONSISTENCY,
                passed=passed,
                confidence=consistency_score,
                severity=ValidationSeverity.MEDIUM if not passed else ValidationSeverity.LOW,
                message=f"Relationship consistency check completed. Issues: {len(issues)}",
                evidence={"issues": issues, "total_relationships": len(response.relationships)},
                suggestions=[f"Review relationship: {issue}" for issue in issues],
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Relationship consistency validation failed: {e}")
            return ValidationResult(
                rule_type=ValidationType.RELATIONSHIP_CONSISTENCY,
                passed=False,
                confidence=0.0,
                severity=ValidationSeverity.HIGH,
                message=f"Validation error: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _is_impossible_relationship(self, source: str, target: str, rel_type: str, context: dict[str, Any]) -> bool:
        """Check if a relationship is logically impossible."""
        # Basic impossibility checks
        if source == target and rel_type in ["parent_of", "child_of", "spouse_of"]:
            return True
        
        # Check temporal impossibility (if we have date information)
        source_date = context.get("entity_dates", {}).get(source)
        target_date = context.get("entity_dates", {}).get(target)
        
        if source_date and target_date:
            try:
                source_year = int(source_date.split("-")[0])
                target_year = int(target_date.split("-")[0])
                
                if rel_type == "parent_of" and source_year > target_year:
                    return True
            except:
                pass
        
        return False
    
    async def validate_temporal_consistency(self, response: RAGResponse, context: dict[str, Any]) -> ValidationResult:
        """Validate temporal order and date consistency."""
        start_time = time.time()
        
        try:
            # Extract dates from content
            date_pattern = r"(\d{4})"
            dates = re.findall(date_pattern, response.content)
            
            if len(dates) < 2:
                return ValidationResult(
                    rule_type=ValidationType.TEMPORAL_CONSISTENCY,
                    passed=True,
                    confidence=1.0,
                    severity=ValidationSeverity.LOW,
                    message="Insufficient temporal information to validate",
                    execution_time=time.time() - start_time
                )
            
            issues = []
            years = [int(date) for date in dates]
            
            # Check for unreasonable year ranges
            min_year, max_year = min(years), max(years)
            year_range = max_year - min_year
            
            if year_range > 100:
                issues.append(f"Large temporal range detected: {min_year}-{max_year}")
            
            if min_year < 1000 or max_year > 2050:
                issues.append(f"Potentially invalid years: {min_year}-{max_year}")
            
            # Check temporal order in relationships
            for rel in response.relationships:
                if "before" in rel.get("type", "").lower() or "after" in rel.get("type", "").lower():
                    # Could implement more sophisticated temporal relationship checking
                    pass
            
            consistency_score = max(0.0, 1.0 - (len(issues) * 0.3))
            passed = consistency_score >= self.validation_rules[ValidationType.TEMPORAL_CONSISTENCY].threshold
            
            return ValidationResult(
                rule_type=ValidationType.TEMPORAL_CONSISTENCY,
                passed=passed,
                confidence=consistency_score,
                severity=ValidationSeverity.HIGH if not passed else ValidationSeverity.LOW,
                message=f"Temporal consistency check completed. Issues: {len(issues)}",
                evidence={"issues": issues, "date_range": f"{min_year}-{max_year}", "total_dates": len(dates)},
                suggestions=[f"Review temporal claim: {issue}" for issue in issues],
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Temporal consistency validation failed: {e}")
            return ValidationResult(
                rule_type=ValidationType.TEMPORAL_CONSISTENCY,
                passed=False,
                confidence=0.0,
                severity=ValidationSeverity.HIGH,
                message=f"Validation error: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    async def validate_factual_consistency(self, response: RAGResponse, context: dict[str, Any]) -> ValidationResult:
        """Validate factual accuracy and logical consistency."""
        start_time = time.time()
        
        try:
            # Use hallucination detector for factual inconsistencies
            inconsistency_result = await self.hallucination_detector.detect_factual_inconsistencies(
                response.content, context
            )
            
            issues = inconsistency_result.get("inconsistencies", [])
            inconsistency_score = inconsistency_result.get("inconsistency_score", 0.0)
            
            # Additional factual checks
            content_lower = response.content.lower()
            
            # Check for contradictory statements
            contradiction_patterns = [
                (r"is.*not", r"is.*"),
                (r"was.*never", r"was.*"),
                (r"cannot.*", r"can.*"),
                (r"impossible.*", r"possible.*")
            ]
            
            for neg_pattern, pos_pattern in contradiction_patterns:
                neg_matches = re.findall(neg_pattern, content_lower)
                pos_matches = re.findall(pos_pattern, content_lower)
                
                if neg_matches and pos_matches:
                    issues.append({
                        "type": "contradiction",
                        "description": "Potentially contradictory statements found",
                        "severity": "medium"
                    })
            
            consistency_score = max(0.0, 1.0 - inconsistency_score)
            passed = consistency_score >= self.validation_rules[ValidationType.FACTUAL_CONSISTENCY].threshold
            
            return ValidationResult(
                rule_type=ValidationType.FACTUAL_CONSISTENCY,
                passed=passed,
                confidence=consistency_score,
                severity=ValidationSeverity.HIGH if not passed else ValidationSeverity.LOW,
                message=f"Factual consistency check completed. Issues: {len(issues)}",
                evidence={"issues": issues, "inconsistency_score": inconsistency_score},
                suggestions=[f"Review factual claim: {issue.get('description', issue)}" for issue in issues],
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Factual consistency validation failed: {e}")
            return ValidationResult(
                rule_type=ValidationType.FACTUAL_CONSISTENCY,
                passed=False,
                confidence=0.0,
                severity=ValidationSeverity.HIGH,
                message=f"Validation error: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    async def validate_hallucination_detection(self, response: RAGResponse, context: dict[str, Any]) -> ValidationResult:
        """Detect potential hallucinations and uncertain statements."""
        start_time = time.time()
        
        try:
            # Detect uncertainty language
            uncertainty_result = await self.hallucination_detector.detect_uncertainty_language(response.content)
            
            uncertainty_score = uncertainty_result.get("uncertainty_score", 0.0)
            uncertainty_matches = uncertainty_result.get("matches", [])
            
            # Calculate confidence (inverse of uncertainty)
            confidence = max(0.0, 1.0 - uncertainty_score)
            passed = uncertainty_score <= self.validation_rules[ValidationType.HALLUCINATION_DETECTION].threshold
            
            suggestions = []
            if uncertainty_matches:
                suggestions.extend([
                    f"Remove uncertainty language: '{match['match']}'" 
                    for match in uncertainty_matches[:3]  # Limit suggestions
                ])
            
            return ValidationResult(
                rule_type=ValidationType.HALLUCINATION_DETECTION,
                passed=passed,
                confidence=confidence,
                severity=ValidationSeverity.MEDIUM if not passed else ValidationSeverity.LOW,
                message=f"Hallucination detection completed. Uncertainty score: {uncertainty_score:.2f}",
                evidence={
                    "uncertainty_score": uncertainty_score,
                    "uncertainty_matches": len(uncertainty_matches),
                    "matches": uncertainty_matches[:5]  # Limit evidence size
                },
                suggestions=suggestions,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Hallucination detection failed: {e}")
            return ValidationResult(
                rule_type=ValidationType.HALLUCINATION_DETECTION,
                passed=False,
                confidence=0.0,
                severity=ValidationSeverity.HIGH,
                message=f"Validation error: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    async def validate_response(self, response: RAGResponse, query: str, context: dict[str, Any]) -> list[ValidationResult]:
        """
        Perform comprehensive validation on a RAG response.
        
        Args:
            response: RAG response to validate
            query: Original query
            context: Additional context for validation
            
        Returns:
            List of validation results
        """
        validation_tasks = []
        
        # Add enabled validation tasks
        for rule_type, rule in self.validation_rules.items():
            if not rule.enabled:
                continue
            
            if rule_type == ValidationType.ENTITY_CONSISTENCY:
                validation_tasks.append(self.validate_entity_consistency(response, context))
            elif rule_type == ValidationType.RELATIONSHIP_CONSISTENCY:
                validation_tasks.append(self.validate_relationship_consistency(response, context))
            elif rule_type == ValidationType.TEMPORAL_CONSISTENCY:
                validation_tasks.append(self.validate_temporal_consistency(response, context))
            elif rule_type == ValidationType.FACTUAL_CONSISTENCY:
                validation_tasks.append(self.validate_factual_consistency(response, context))
            elif rule_type == ValidationType.HALLUCINATION_DETECTION:
                validation_tasks.append(self.validate_hallucination_detection(response, context))
        
        # Execute all validations in parallel
        if validation_tasks:
            validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Filter out exceptions and log them
            valid_results = []
            for result in validation_results:
                if isinstance(result, Exception):
                    logger.error(f"Validation task failed: {result}")
                else:
                    valid_results.append(result)
            
            return valid_results
        
        return []


class AutomatedCorrector:
    """Automated correction mechanism for validation issues."""
    
    def __init__(self):
        self.correction_strategies = {
            ValidationType.ENTITY_CONSISTENCY: self._correct_entity_consistency,
            ValidationType.RELATIONSHIP_CONSISTENCY: self._correct_relationship_consistency,
            ValidationType.TEMPORAL_CONSISTENCY: self._correct_temporal_consistency,
            ValidationType.FACTUAL_CONSISTENCY: self._correct_factual_consistency,
            ValidationType.HALLUCINATION_DETECTION: self._correct_hallucinations
        }
    
    async def _correct_entity_consistency(self, response: RAGResponse, validation_result: ValidationResult) -> CorrectionSuggestion:
        """Correct entity consistency issues."""
        issues = validation_result.evidence.get("issues", [])
        
        if not issues:
            return CorrectionSuggestion(
                action=CorrectionAction.ACCEPT,
                confidence=1.0,
                original_text=response.content,
                corrected_text=response.content,
                reasoning="No entity consistency issues found"
            )
        
        # Simple entity standardization
        corrected_content = response.content
        
        for issue in issues:
            if "Entity variations found:" in issue:
                # Extract entity variations and standardize to the first one
                entities_part = issue.split(": ")[1]
                entities = [e.strip() for e in entities_part.split(",")]
                
                if len(entities) > 1:
                    primary_entity = entities[0]
                    for variant in entities[1:]:
                        corrected_content = corrected_content.replace(variant, primary_entity)
        
        return CorrectionSuggestion(
            action=CorrectionAction.MODIFY,
            confidence=0.7,
            original_text=response.content,
            corrected_text=corrected_content,
            reasoning=f"Standardized entity names to resolve {len(issues)} consistency issues"
        )
    
    async def _correct_relationship_consistency(self, response: RAGResponse, validation_result: ValidationResult) -> CorrectionSuggestion:
        """Correct relationship consistency issues."""
        issues = validation_result.evidence.get("issues", [])
        
        if not issues:
            return CorrectionSuggestion(
                action=CorrectionAction.ACCEPT,
                confidence=1.0,
                original_text=response.content,
                corrected_text=response.content,
                reasoning="No relationship consistency issues found"
            )
        
        # For complex relationship issues, flag for review rather than auto-correct
        return CorrectionSuggestion(
            action=CorrectionAction.FLAG_UNCERTAIN,
            confidence=0.5,
            original_text=response.content,
            corrected_text=response.content,
            reasoning=f"Relationship consistency issues detected: {issues[:2]}. Manual review recommended."
        )
    
    async def _correct_temporal_consistency(self, response: RAGResponse, validation_result: ValidationResult) -> CorrectionSuggestion:
        """Correct temporal consistency issues."""
        issues = validation_result.evidence.get("issues", [])
        
        if not issues:
            return CorrectionSuggestion(
                action=CorrectionAction.ACCEPT,
                confidence=1.0,
                original_text=response.content,
                corrected_text=response.content,
                reasoning="No temporal consistency issues found"
            )
        
        # Temporal issues often require domain knowledge, so flag for review
        return CorrectionSuggestion(
            action=CorrectionAction.FLAG_UNCERTAIN,
            confidence=0.3,
            original_text=response.content,
            corrected_text=response.content,
            reasoning=f"Temporal consistency issues detected: {issues}. Manual verification required."
        )
    
    async def _correct_factual_consistency(self, response: RAGResponse, validation_result: ValidationResult) -> CorrectionSuggestion:
        """Correct factual consistency issues."""
        issues = validation_result.evidence.get("issues", [])
        
        if not issues:
            return CorrectionSuggestion(
                action=CorrectionAction.ACCEPT,
                confidence=1.0,
                original_text=response.content,
                corrected_text=response.content,
                reasoning="No factual consistency issues found"
            )
        
        # Factual errors require careful verification, recommend regeneration
        return CorrectionSuggestion(
            action=CorrectionAction.REGENERATE,
            confidence=0.4,
            original_text=response.content,
            corrected_text="",
            reasoning=f"Factual consistency issues detected: {len(issues)} issues. Regeneration recommended."
        )
    
    async def _correct_hallucinations(self, response: RAGResponse, validation_result: ValidationResult) -> CorrectionSuggestion:
        """Correct hallucination issues by removing uncertainty language."""
        uncertainty_score = validation_result.evidence.get("uncertainty_score", 0.0)
        matches = validation_result.evidence.get("matches", [])
        
        if uncertainty_score < 0.2:
            return CorrectionSuggestion(
                action=CorrectionAction.ACCEPT,
                confidence=1.0,
                original_text=response.content,
                corrected_text=response.content,
                reasoning="Low uncertainty score, no corrections needed"
            )
        
        # Remove uncertainty language
        corrected_content = response.content
        
        uncertainty_phrases = [
            "I think", "I believe", "I assume", "It seems", "probably", "maybe", "possibly",
            "as far as I know", "to my knowledge", "I'm not certain",
            "could be", "might be", "may be", "it's possible"
        ]
        
        for phrase in uncertainty_phrases:
            # Remove the phrases but be careful with sentence structure
            corrected_content = re.sub(rf'\b{re.escape(phrase)}\b\s*', '', corrected_content, flags=re.IGNORECASE)
        
        # Clean up any double spaces
        corrected_content = re.sub(r'\s+', ' ', corrected_content).strip()
        
        return CorrectionSuggestion(
            action=CorrectionAction.MODIFY,
            confidence=0.8,
            original_text=response.content,
            corrected_text=corrected_content,
            reasoning=f"Removed uncertainty language. {len(matches)} uncertain phrases addressed."
        )
    
    async def generate_corrections(self, response: RAGResponse, validation_results: list[ValidationResult]) -> list[CorrectionSuggestion]:
        """
        Generate correction suggestions based on validation results.
        
        Args:
            response: Original RAG response
            validation_results: List of validation results
            
        Returns:
            List of correction suggestions
        """
        correction_tasks = []
        
        for validation_result in validation_results:
            if not validation_result.passed and validation_result.rule_type in self.correction_strategies:
                correction_strategy = self.correction_strategies[validation_result.rule_type]
                correction_tasks.append(correction_strategy(response, validation_result))
        
        if correction_tasks:
            corrections = await asyncio.gather(*correction_tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_corrections = []
            for correction in corrections:
                if isinstance(correction, Exception):
                    logger.error(f"Correction generation failed: {correction}")
                else:
                    valid_corrections.append(correction)
            
            return valid_corrections
        
        return []


class SelfCorrectionFramework:
    """
    Main self-correction framework that orchestrates validation and correction.
    """
    
    def __init__(self):
        self.validation_engine = ValidationEngine()
        self.confidence_scorer = ConfidenceScorer()
        self.automated_corrector = AutomatedCorrector()
        
        self.enabled = Config.SELF_CORRECTION_ENABLED
        self.min_confidence = Config.VALIDATION_MIN_CONFIDENCE
        self.max_iterations = Config.VALIDATION_MAX_ITERATIONS
        self.timeout_seconds = Config.VALIDATION_TIMEOUT_SECONDS
    
    async def validate_and_correct(self, response: RAGResponse, query: str, context: dict[str, Any]) -> RAGResponse:
        """
        Perform comprehensive validation and correction on a RAG response.
        
        Args:
            response: RAG response to validate and correct
            query: Original query
            context: Additional context for validation
            
        Returns:
            Enhanced RAG response with validation results and corrections
        """
        if not self.enabled:
            logger.info("Self-correction framework is disabled")
            return response
        
        start_time = time.time()
        iteration = 0
        current_response = response
        
        try:
            while iteration < self.max_iterations:
                # Check timeout
                if time.time() - start_time > self.timeout_seconds:
                    logger.warning(f"Self-correction timeout after {iteration} iterations")
                    break
                
                iteration += 1
                logger.debug(f"Self-correction iteration {iteration}")
                
                # Perform validation
                validation_results = await self.validation_engine.validate_response(
                    current_response, query, context
                )
                
                # Calculate confidence score
                confidence_score = await self.confidence_scorer.calculate_confidence_score(
                    current_response, query, context
                )
                
                # Update response with validation results
                current_response.validation_results = validation_results
                current_response.confidence_score = confidence_score
                
                # Check if validation passed
                failed_validations = [vr for vr in validation_results if not vr.passed]
                
                if not failed_validations and confidence_score.overall_confidence >= self.min_confidence:
                    logger.info(f"Validation passed after {iteration} iterations")
                    break
                
                # Generate corrections
                corrections = await self.automated_corrector.generate_corrections(
                    current_response, failed_validations
                )
                
                current_response.correction_suggestions = corrections
                
                # Apply automatic corrections if available
                best_correction = self._select_best_correction(corrections)
                
                if best_correction and best_correction.action == CorrectionAction.MODIFY:
                    # Apply the correction
                    current_response.content = best_correction.corrected_text
                    logger.debug(f"Applied correction: {best_correction.reasoning}")
                elif best_correction and best_correction.action in [CorrectionAction.REGENERATE, CorrectionAction.REJECT]:
                    # These actions require external intervention
                    logger.info(f"Correction requires external action: {best_correction.action}")
                    break
                else:
                    # No actionable corrections available
                    logger.info("No actionable corrections available, stopping iterations")
                    break
            
            # Add metadata about the correction process
            current_response.metadata.update({
                "self_correction": {
                    "enabled": True,
                    "iterations": iteration,
                    "total_validations": len(validation_results),
                    "failed_validations": len([vr for vr in validation_results if not vr.passed]),
                    "final_confidence": confidence_score.overall_confidence,
                    "processing_time": time.time() - start_time,
                    "corrections_applied": len([c for c in current_response.correction_suggestions 
                                               if c.action == CorrectionAction.MODIFY])
                }
            })
            
            logger.info(f"Self-correction completed in {iteration} iterations. "
                       f"Final confidence: {confidence_score.overall_confidence:.3f}")
            
            return current_response
            
        except Exception as e:
            logger.error(f"Self-correction framework error: {e}")
            # Return original response with error information
            response.metadata["self_correction_error"] = str(e)
            return response
    
    def _select_best_correction(self, corrections: list[CorrectionSuggestion]) -> Optional[CorrectionSuggestion]:
        """
        Select the best correction suggestion based on confidence and action type.
        
        Args:
            corrections: List of correction suggestions
            
        Returns:
            Best correction suggestion or None
        """
        if not corrections:
            return None
        
        # Prefer corrections in this order: MODIFY, FLAG_UNCERTAIN, REGENERATE, REJECT
        action_priority = {
            CorrectionAction.MODIFY: 1,
            CorrectionAction.FLAG_UNCERTAIN: 2,
            CorrectionAction.REGENERATE: 3,
            CorrectionAction.REJECT: 4,
            CorrectionAction.ACCEPT: 5
        }
        
        # Sort by action priority and confidence
        sorted_corrections = sorted(
            corrections,
            key=lambda c: (action_priority.get(c.action, 10), -c.confidence)
        )
        
        return sorted_corrections[0]
    
    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on the self-correction framework.
        
        Returns:
            Health status information
        """
        try:
            # Test basic functionality
            test_response = RAGResponse(
                content="This is a test response for health checking.",
                entities=["test"],
                relationships=[{"source": "test", "target": "health", "type": "checks"}]
            )
            
            test_context = {"test": True}
            start_time = time.time()
            
            # Run a quick validation
            validation_results = await self.validation_engine.validate_response(
                test_response, "test query", test_context
            )
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "enabled": self.enabled,
                "response_time": response_time,
                "validation_rules": len(self.validation_engine.validation_rules),
                "enabled_rules": sum(1 for rule in self.validation_engine.validation_rules.values() if rule.enabled),
                "test_validations": len(validation_results),
                "configuration": {
                    "min_confidence": self.min_confidence,
                    "max_iterations": self.max_iterations,
                    "timeout_seconds": self.timeout_seconds
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "enabled": self.enabled
            }