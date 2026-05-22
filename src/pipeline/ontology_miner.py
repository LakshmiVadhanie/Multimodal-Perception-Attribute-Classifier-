"""
Ontology-guided prompt construction for VLM auto-labeling.

Builds structured prompts that guide the VLM to produce labels
consistent with the defined attribute ontology.
"""

from typing import Dict, List, Optional

from src.dataset.ontology import ONTOLOGY, AttributeValue, Attribute, get_applicable_attributes


class OntologyMiner:
    """
    Constructs VLM prompts from the attribute ontology.

    For each attribute, builds a prompt that:
    1. Describes what to look for
    2. Lists valid label options with descriptions
    3. Asks the model to pick exactly one
    """

    def __init__(self, use_chain_of_thought: bool = True):
        self.use_chain_of_thought = use_chain_of_thought

    def build_prompt(self, attribute_name: str, road_user_type: str) -> str:
        attr = ONTOLOGY[attribute_name]
        options_text = self._format_options(attr.values)

        if self.use_chain_of_thought:
            prompt = (
                f"You are analyzing an image of a {road_user_type} on a road.\n\n"
                f"Task: {attr.vlm_question}\n\n"
                f"Think step by step about what you observe, then choose exactly one "
                f"of the following options:\n\n"
                f"{options_text}\n\n"
                f"First briefly describe what you see, then on the last line output "
                f"only the label in this exact format:\n"
                f"LABEL: <chosen_label>"
            )
        else:
            prompt = (
                f"You are analyzing an image of a {road_user_type}.\n"
                f"{attr.vlm_question}\n\n"
                f"Choose exactly one option:\n{options_text}\n\n"
                f"Output only the label:\nLABEL: <chosen_label>"
            )

        return prompt

    def build_batch_prompt(self, road_user_type: str) -> str:
        """
        Single prompt that asks the VLM to label all applicable attributes at once.
        More efficient but may be less accurate than per-attribute prompting.
        """
        applicable = get_applicable_attributes(road_user_type)
        sections = []
        for attr_name in applicable:
            attr = ONTOLOGY[attr_name]
            options_text = self._format_options(attr.values)
            sections.append(
                f"[{attr.display_name}]\n"
                f"Question: {attr.vlm_question}\n"
                f"Options:\n{options_text}"
            )

        all_sections = "\n\n".join(sections)
        label_lines = "\n".join(
            f"{name}: <label>" for name in applicable
        )

        return (
            f"You are analyzing an image of a {road_user_type} on a road.\n\n"
            f"Answer each of the following questions about the {road_user_type}:\n\n"
            f"{all_sections}\n\n"
            f"Output your answers in this exact format (one per line):\n"
            f"{label_lines}"
        )

    def _format_options(self, values: List[AttributeValue]) -> str:
        lines = []
        for v in values:
            lines.append(f"  - {v.label}: {v.description}")
        return "\n".join(lines)

    def parse_single_label_response(self, response: str, attribute_name: str) -> Optional[str]:
        """
        Extract the label from a VLM response for a single-attribute prompt.
        Returns None if parsing fails.
        """
        attr = ONTOLOGY[attribute_name]
        valid_labels = {v.label for v in attr.values}
        valid_keywords = {
            keyword: v.label
            for v in attr.values
            for keyword in v.vlm_keywords
        }

        # look for "LABEL: <label>" format first
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("LABEL:"):
                candidate = line.split(":", 1)[1].strip().lower().replace(" ", "_")
                if candidate in valid_labels:
                    return candidate

        # fallback: scan for valid label strings in the response
        response_lower = response.lower()
        for label in valid_labels:
            if label.replace("_", " ") in response_lower or label in response_lower:
                return label

        # fallback: scan for known VLM keywords
        for keyword, label in valid_keywords.items():
            if keyword.lower() in response_lower:
                return label

        return None

    def parse_batch_label_response(
        self, response: str, road_user_type: str
    ) -> Dict[str, Optional[str]]:
        """
        Parse a batch VLM response covering all attributes.
        Returns a dict mapping attribute_name -> label (or None).
        """
        applicable = get_applicable_attributes(road_user_type)
        results = {name: None for name in applicable}

        for line in response.strip().split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip().lower().replace(" ", "_")
            value = value.strip().lower().replace(" ", "_")

            if key in results:
                attr = ONTOLOGY[key]
                valid = {v.label for v in attr.values}
                if value in valid:
                    results[key] = value

        return results
