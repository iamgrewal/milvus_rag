"""
Configuration Validator for Phase 2 Hybrid RAG System

Provides validation and reporting tools for the enhanced configuration system.
"""

import json
from typing import Any

from graphrag.config.settings import Config
from graphrag.logger import logger


class ConfigurationValidator:
    """
    Validates and reports on configuration settings for the hybrid RAG system.
    """

    @staticmethod
    def validate_all() -> dict[str, Any]:
        """
        Perform comprehensive configuration validation.

        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "phase2_features": [],
            "discovery_strategies": [],
            "summary": {},
        }

        try:
            # Run built-in validation
            errors = Config.validate_configuration()
            validation_result["errors"] = errors
            validation_result["valid"] = len(errors) == 0

            # Get enabled features
            validation_result["phase2_features"] = Config.get_phase2_enabled_features()
            validation_result["discovery_strategies"] = (
                Config.get_discovery_strategies()
            )

            # Additional validation checks
            warnings = ConfigurationValidator._check_performance_warnings()
            validation_result["warnings"] = warnings

            # Generate summary
            validation_result["summary"] = ConfigurationValidator._generate_summary()

            return validation_result

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
            return validation_result

    @staticmethod
    def _check_performance_warnings() -> list[str]:
        """
        Check for potential performance issues in configuration.

        Returns:
            List of performance warnings
        """
        warnings = []

        # Check cache sizes
        if Config.MEMORY_MAX_CACHE_SIZE > 50000:
            warnings.append(
                "MEMORY_MAX_CACHE_SIZE is very high, may consume excessive memory"
            )

        if Config.EMBEDDING_CACHE_MAX_SIZE > 100000:
            warnings.append(
                "EMBEDDING_CACHE_MAX_SIZE is very high, may consume excessive memory"
            )

        # Check timeout values
        if Config.MULTIHOP_TIMEOUT_SECONDS > 60:
            warnings.append("MULTIHOP_TIMEOUT_SECONDS is high, may cause slow queries")

        if Config.VALIDATION_TIMEOUT_SECONDS > 120:
            warnings.append(
                "VALIDATION_TIMEOUT_SECONDS is high, may cause slow responses"
            )

        # Check batch sizes
        if Config.BATCH_SIZE_ENTITIES > 5000:
            warnings.append("BATCH_SIZE_ENTITIES is very high, may cause memory issues")

        # Check similarity thresholds
        if Config.VECTOR_SIMILARITY_THRESHOLD < 0.3:
            warnings.append(
                "VECTOR_SIMILARITY_THRESHOLD is low, may return irrelevant results"
            )

        if Config.GRAPH_SIMILARITY_THRESHOLD < 0.2:
            warnings.append(
                "GRAPH_SIMILARITY_THRESHOLD is low, may return weak relationships"
            )

        return warnings

    @staticmethod
    def _generate_summary() -> dict[str, Any]:
        """
        Generate configuration summary.

        Returns:
            Dictionary with configuration summary
        """
        return {
            "phase2_enabled": len(Config.get_phase2_enabled_features()) > 0,
            "total_discovery_strategies": len(Config.get_discovery_strategies()),
            "caching_enabled": {
                "query_cache": Config.QUERY_CACHE_ENABLED,
                "embedding_cache": Config.EMBEDDING_CACHE_ENABLED,
                "entity_cache": Config.ENTITY_CACHE_ENABLED,
                "memory_system": Config.MEMORY_SYSTEM_ENABLED,
            },
            "validation_enabled": {
                "self_correction": Config.SELF_CORRECTION_ENABLED,
                "confidence_scoring": Config.CONFIDENCE_SCORING_ENABLED,
                "hallucination_detection": Config.HALLUCINATION_DETECTION_ENABLED,
            },
            "performance_features": {
                "monitoring": Config.PERFORMANCE_MONITORING_ENABLED,
                "pattern_discovery": Config.PATTERN_DISCOVERY_ENABLED,
                "graph_expansion": Config.GRAPH_EXPANSION_ENABLED,
            },
            "hybrid_weights": {
                "vector_weight": Config.HYBRID_VECTOR_WEIGHT,
                "graph_weight": Config.HYBRID_GRAPH_WEIGHT,
            },
        }

    @staticmethod
    def print_validation_report(validation_result: dict[str, Any]) -> None:
        """
        Print a formatted validation report.

        Args:
            validation_result: Result from validate_all()
        """
        print("=" * 60)
        print("Phase 2 Hybrid RAG Configuration Validation Report")
        print("=" * 60)

        # Overall status
        status = "✅ VALID" if validation_result["valid"] else "❌ INVALID"
        print(f"\nOverall Status: {status}")

        # Errors
        if validation_result["errors"]:
            print(f"\n🚨 Errors ({len(validation_result['errors'])}):")
            for i, error in enumerate(validation_result["errors"], 1):
                print(f"  {i}. {error}")

        # Warnings
        if validation_result["warnings"]:
            print(f"\n⚠️  Warnings ({len(validation_result['warnings'])}):")
            for i, warning in enumerate(validation_result["warnings"], 1):
                print(f"  {i}. {warning}")

        # Enabled Features
        print(
            f"\n🚀 Enabled Phase 2 Features ({len(validation_result['phase2_features'])}):"
        )
        for feature in validation_result["phase2_features"]:
            print(f"  • {feature}")

        # Discovery Strategies
        print(
            f"\n🔍 Active Discovery Strategies ({len(validation_result['discovery_strategies'])}):"
        )
        for strategy in validation_result["discovery_strategies"]:
            print(f"  • {strategy}")

        # Summary
        summary = validation_result["summary"]
        print("\n📊 Configuration Summary:")
        print(f"  • Phase 2 Enabled: {summary['phase2_enabled']}")
        print(f"  • Discovery Strategies: {summary['total_discovery_strategies']}/4")
        print(
            f"  • Caching Systems: {sum(summary['caching_enabled'].values())}/4 enabled"
        )
        print(
            f"  • Validation Features: {sum(summary['validation_enabled'].values())}/3 enabled"
        )
        print(
            f"  • Performance Features: {sum(summary['performance_features'].values())}/3 enabled"
        )
        print(
            f"  • Hybrid Weights: Vector={summary['hybrid_weights']['vector_weight']}, Graph={summary['hybrid_weights']['graph_weight']}"
        )

        print("\n" + "=" * 60)

    @staticmethod
    def export_configuration(file_path: str = "config_export.json") -> bool:
        """
        Export current configuration to JSON file.

        Args:
            file_path: Path to export file

        Returns:
            True if successful, False otherwise
        """
        try:
            config_dict = {}

            # Get all configuration attributes
            for attr_name in dir(Config):
                if not attr_name.startswith("_") and not callable(
                    getattr(Config, attr_name)
                ):
                    config_dict[attr_name] = getattr(Config, attr_name)

            # Add metadata
            config_export = {
                "metadata": {
                    "export_type": "Phase 2 Hybrid RAG Configuration",
                    "validation_status": (
                        "valid"
                        if len(Config.validate_configuration()) == 0
                        else "invalid"
                    ),
                    "phase2_features": Config.get_phase2_enabled_features(),
                    "discovery_strategies": Config.get_discovery_strategies(),
                },
                "configuration": config_dict,
            }

            with open(file_path, "w") as f:
                json.dump(config_export, f, indent=2, default=str)

            logger.info(f"Configuration exported to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False

    @staticmethod
    def check_readiness() -> dict[str, bool]:
        """
        Check if system is ready for Phase 2 operations.

        Returns:
            Dictionary with readiness status for each component
        """
        readiness = {
            "configuration_valid": len(Config.validate_configuration()) == 0,
            "context_enhancement_ready": (
                Config.CONTEXT_ENHANCEMENT_ENABLED
                and len(Config.get_discovery_strategies()) > 0
            ),
            "memory_system_ready": (
                Config.MEMORY_SYSTEM_ENABLED
                and Config.REDIS_HOST
                and Config.MEMORY_MAX_CACHE_SIZE > 0
            ),
            "validation_ready": (
                Config.SELF_CORRECTION_ENABLED and Config.CONFIDENCE_SCORING_ENABLED
            ),
            "performance_monitoring_ready": (
                Config.PERFORMANCE_MONITORING_ENABLED and Config.HEALTH_CHECK_ENABLED
            ),
            "hybrid_fusion_ready": (
                abs((Config.HYBRID_VECTOR_WEIGHT + Config.HYBRID_GRAPH_WEIGHT) - 1.0)
                < 0.01
            ),
        }

        readiness["overall_ready"] = all(readiness.values())

        return readiness


def main():
    """Main function for command-line usage."""
    print("Validating Phase 2 Hybrid RAG Configuration...")

    # Perform validation
    validation_result = ConfigurationValidator.validate_all()

    # Print report
    ConfigurationValidator.print_validation_report(validation_result)

    # Check readiness
    readiness = ConfigurationValidator.check_readiness()
    print("\n🔧 System Readiness:")
    for component, ready in readiness.items():
        status = "✅" if ready else "❌"
        print(f"  {status} {component.replace('_', ' ').title()}")

    # Export configuration
    if validation_result["valid"]:
        ConfigurationValidator.export_configuration()
        print("\n💾 Configuration exported to config_export.json")


if __name__ == "__main__":
    main()
