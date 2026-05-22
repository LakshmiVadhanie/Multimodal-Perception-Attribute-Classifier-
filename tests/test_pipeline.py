"""Tests for the VLM auto-labeling pipeline components."""

import pytest
from src.pipeline.ontology_miner import OntologyMiner
from src.pipeline.label_validator import LabelValidator
from src.dataset.ontology import ONTOLOGY, get_applicable_attributes


class TestOntologyMiner:
    def setup_method(self):
        self.miner = OntologyMiner(use_chain_of_thought=True)

    def test_build_prompt_contains_question(self):
        prompt = self.miner.build_prompt("mobility", "vehicle")
        assert "moving" in prompt.lower() or "stationary" in prompt.lower()

    def test_build_prompt_contains_label_options(self):
        prompt = self.miner.build_prompt("mobility", "vehicle")
        assert "moving" in prompt
        assert "stationary" in prompt
        assert "slow_moving" in prompt

    def test_build_batch_prompt_covers_applicable_attrs(self):
        prompt = self.miner.build_batch_prompt("pedestrian")
        applicable = get_applicable_attributes("pedestrian")
        for attr_name in applicable:
            assert attr_name in prompt

    def test_parse_label_format(self):
        response = "The vehicle appears to be moving quickly.\nLABEL: moving"
        result = self.miner.parse_single_label_response(response, "mobility")
        assert result == "moving"

    def test_parse_label_case_insensitive(self):
        response = "LABEL: MOVING"
        result = self.miner.parse_single_label_response(response, "mobility")
        assert result == "moving"

    def test_parse_label_fallback_to_keyword_scan(self):
        response = "The road user is clearly stopped and not moving."
        result = self.miner.parse_single_label_response(response, "mobility")
        assert result == "stationary"

    def test_parse_invalid_label_returns_none(self):
        response = "LABEL: flying"
        result = self.miner.parse_single_label_response(response, "mobility")
        assert result is None

    def test_parse_batch_response(self):
        response = "mobility: moving\nocclusion: none\nlighting: well_lit"
        results = self.miner.parse_batch_label_response(response, "vehicle")
        assert results.get("mobility") == "moving"
        assert results.get("occlusion") == "none"
        assert results.get("lighting") == "well_lit"

    def test_parse_batch_invalid_value_is_none(self):
        response = "mobility: flying\nocclusion: none"
        results = self.miner.parse_batch_label_response(response, "vehicle")
        assert results.get("mobility") is None
        assert results.get("occlusion") == "none"


class TestLabelValidator:
    def setup_method(self):
        self.validator = LabelValidator(confidence_threshold=0.75)

    def test_valid_labels_pass_through(self):
        labels = {
            "mobility": "moving",
            "occlusion": "none",
            "lighting": "well_lit",
        }
        result = self.validator.validate(labels, "vehicle")
        assert result["mobility"] == "moving"
        assert result["occlusion"] == "none"
        assert result["lighting"] == "well_lit"

    def test_invalid_label_value_set_to_none(self):
        labels = {"mobility": "flying"}
        result = self.validator.validate(labels, "vehicle")
        assert result["mobility"] is None

    def test_non_applicable_attribute_set_to_none(self):
        # attention is only for pedestrians
        labels = {"attention": "attentive", "mobility": "moving"}
        result = self.validator.validate(labels, "vehicle")
        assert result.get("attention") is None
        assert result["mobility"] == "moving"

    def test_none_labels_stay_none(self):
        labels = {"mobility": None, "occlusion": "none"}
        result = self.validator.validate(labels, "vehicle")
        assert result["mobility"] is None
        assert result["occlusion"] == "none"

    def test_coverage_report(self):
        annotations = [
            {
                "road_user_type": "vehicle",
                "attributes": {"mobility": "moving", "occlusion": "none"},
            },
            {
                "road_user_type": "vehicle",
                "attributes": {"mobility": "stationary", "occlusion": None},
            },
        ]
        coverage = self.validator.compute_coverage(annotations)
        assert "mobility" in coverage
        assert coverage["mobility"] == pytest.approx(1.0)
        assert coverage["occlusion"] == pytest.approx(0.5)

    def test_consistency_rule_heavy_occlusion_disallows_large_size(self):
        labels = {
            "occlusion": "heavy",
            "size": "large",  # should be rejected by consistency rule
            "mobility": "moving",
        }
        result = self.validator.validate(labels, "vehicle")
        assert result["size"] is None
        assert result["mobility"] == "moving"

    def test_consistency_rule_allows_valid_combination(self):
        labels = {
            "occlusion": "heavy",
            "size": "small",  # valid for heavily occluded
        }
        result = self.validator.validate(labels, "vehicle")
        assert result["size"] == "small"
