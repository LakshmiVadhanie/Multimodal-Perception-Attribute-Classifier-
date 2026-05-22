"""
Label quality validation for VLM-generated annotations.

Filters out low-confidence or inconsistent labels before they are
added to the training set.
"""

from typing import Dict, List, Optional, Set

from loguru import logger

from src.dataset.ontology import ONTOLOGY, get_applicable_attributes


class LabelValidator:
    """
    Validates and filters VLM-generated attribute labels.

    Checks:
    - Label belongs to the valid set for that attribute
    - Attribute is applicable to the given road user type
    - Cross-attribute consistency rules
    """

    def __init__(self, confidence_threshold: float = 0.75):
        # threshold is a placeholder for when VLMs output confidence scores
        self.confidence_threshold = confidence_threshold

        # consistency rules: if attribute A has value X, attribute B must be one of Y
        self.consistency_rules = [
            # a heavily occluded road user should be small or medium in apparent size
            {
                "if": ("occlusion", "heavy"),
                "then_attr": "size",
                "allowed_values": {"small", "medium"},
            },
            # phone use only makes sense if attention is already set to phone_use
            # (this is basically a self-consistency check)
            {
                "if": ("attention", "phone_use"),
                "then_attr": "attention",
                "allowed_values": {"phone_use"},
            },
        ]

    def validate(
        self,
        labels: Dict[str, Optional[str]],
        road_user_type: str,
    ) -> Dict[str, Optional[str]]:
        """
        Returns a cleaned copy of the labels dict with invalid entries set to None.
        """
        applicable = set(get_applicable_attributes(road_user_type))
        cleaned = {}

        for attr_name, label in labels.items():
            if attr_name not in applicable:
                # attribute does not apply to this road user type
                cleaned[attr_name] = None
                continue

            if label is None:
                cleaned[attr_name] = None
                continue

            valid_labels = {v.label for v in ONTOLOGY[attr_name].values}
            if label not in valid_labels:
                logger.debug(
                    f"Invalid label '{label}' for attribute '{attr_name}'. Dropping."
                )
                cleaned[attr_name] = None
            else:
                cleaned[attr_name] = label

        # apply cross-attribute consistency checks
        cleaned = self._apply_consistency_rules(cleaned)

        return cleaned

    def _apply_consistency_rules(
        self, labels: Dict[str, Optional[str]]
    ) -> Dict[str, Optional[str]]:
        result = dict(labels)
        for rule in self.consistency_rules:
            trigger_attr, trigger_val = rule["if"]
            if labels.get(trigger_attr) == trigger_val:
                target_attr = rule["then_attr"]
                if target_attr in result and result[target_attr] not in rule["allowed_values"]:
                    logger.debug(
                        f"Consistency rule violation: {trigger_attr}={trigger_val} "
                        f"requires {target_attr} in {rule['allowed_values']}, "
                        f"but got {result[target_attr]}. Setting to None."
                    )
                    result[target_attr] = None
        return result

    def compute_coverage(self, annotations: List[Dict]) -> Dict[str, float]:
        """
        Returns the fraction of annotations that have a valid (non-None) label
        for each attribute. Useful for monitoring auto-labeling quality.
        """
        counts = {name: 0 for name in ONTOLOGY}
        totals = {name: 0 for name in ONTOLOGY}

        for ann in annotations:
            road_user_type = ann.get("road_user_type", "vehicle")
            applicable = get_applicable_attributes(road_user_type)
            attrs = ann.get("attributes", {})

            for attr_name in applicable:
                totals[attr_name] += 1
                if attrs.get(attr_name) is not None:
                    counts[attr_name] += 1

        return {
            name: counts[name] / totals[name] if totals[name] > 0 else 0.0
            for name in ONTOLOGY
        }
