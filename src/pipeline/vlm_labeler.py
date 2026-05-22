"""
VLM-assisted auto-labeling pipeline.

Uses BLIP-2 or LLaVA to generate attribute labels for unlabeled road user images.
The pipeline processes images in batches and saves progress checkpoints so it
can be resumed if interrupted.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from loguru import logger
from tqdm import tqdm

from src.pipeline.ontology_miner import OntologyMiner
from src.pipeline.label_validator import LabelValidator
from src.dataset.ontology import get_applicable_attributes


class VLMLabeler:
    """
    Runs a VLM over unlabeled images to generate attribute annotations.

    Supports two VLMs:
      - BLIP-2 (Salesforce/blip2-opt-2.7b) - faster, lower VRAM
      - LLaVA  (llava-hf/llava-1.5-7b-hf)  - slower, more capable

    Uses batch prompting (all attributes in one call) for efficiency.
    Falls back to per-attribute prompting if batch parsing fails.
    """

    def __init__(
        self,
        vlm_model: str = "blip2",
        model_id: Optional[str] = None,
        device: str = "cuda",
        max_new_tokens: int = 128,
        temperature: float = 0.3,
        confidence_threshold: float = 0.75,
    ):
        self.vlm_model = vlm_model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold

        self.miner = OntologyMiner(use_chain_of_thought=True)
        self.validator = LabelValidator(confidence_threshold=confidence_threshold)

        self.model = None
        self.processor = None
        self._model_id = model_id or self._default_model_id()

    def _default_model_id(self) -> str:
        if self.vlm_model == "blip2":
            return "Salesforce/blip2-opt-2.7b"
        elif self.vlm_model == "llava":
            return "llava-hf/llava-1.5-7b-hf"
        raise ValueError(f"Unknown VLM model: {self.vlm_model}. Use 'blip2' or 'llava'.")

    def load_model(self):
        logger.info(f"Loading {self.vlm_model} model: {self._model_id}")
        if self.vlm_model == "blip2":
            self._load_blip2()
        elif self.vlm_model == "llava":
            self._load_llava()
        logger.info("VLM model loaded")

    def _load_blip2(self):
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        self.processor = Blip2Processor.from_pretrained(self._model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self._model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

    def _load_llava(self):
        from transformers import LlavaProcessor, LlavaForConditionalGeneration

        self.processor = LlavaProcessor.from_pretrained(self._model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self._model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

    def label_image(self, image: Image.Image, road_user_type: str) -> Dict[str, Optional[str]]:
        """
        Label all applicable attributes for a single image.
        Returns dict mapping attribute_name -> label string (or None).
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before labeling")

        prompt = self.miner.build_batch_prompt(road_user_type)
        response = self._generate(image, prompt)
        labels = self.miner.parse_batch_label_response(response, road_user_type)

        # for any attribute that failed batch parsing, try individual prompts
        failed = [name for name, val in labels.items() if val is None]
        for attr_name in failed:
            single_prompt = self.miner.build_prompt(attr_name, road_user_type)
            single_response = self._generate(image, single_prompt)
            parsed = self.miner.parse_single_label_response(single_response, attr_name)
            labels[attr_name] = parsed

        return labels

    def _generate(self, image: Image.Image, prompt: str) -> str:
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True)

    def run_pipeline(
        self,
        image_paths: List[str],
        road_user_types: List[str],
        output_file: str,
        checkpoint_file: Optional[str] = None,
        save_every_n: int = 100,
    ) -> List[Dict]:
        """
        Label a list of images and save results to a JSON file.

        Checkpointing ensures progress is not lost on interruption.
        """
        if self.model is None:
            self.load_model()

        results = []
        processed_paths = set()

        # resume from checkpoint if available
        if checkpoint_file and Path(checkpoint_file).exists():
            with open(checkpoint_file) as f:
                results = json.load(f)
            processed_paths = {r["image_path"] for r in results}
            logger.info(f"Resuming from checkpoint: {len(results)} already labeled")

        pending = [
            (path, rtype)
            for path, rtype in zip(image_paths, road_user_types)
            if path not in processed_paths
        ]

        logger.info(f"Labeling {len(pending)} images")

        for i, (image_path, road_user_type) in enumerate(tqdm(pending, desc="Auto-labeling")):
            try:
                image = Image.open(image_path).convert("RGB")
                labels = self.label_image(image, road_user_type)
                validated = self.validator.validate(labels, road_user_type)

                results.append({
                    "image_path": image_path,
                    "road_user_type": road_user_type,
                    "attributes": validated,
                    "labeling_method": "vlm_auto",
                    "vlm_model": self.vlm_model,
                })
            except Exception as e:
                logger.warning(f"Failed to label {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "road_user_type": road_user_type,
                    "attributes": {},
                    "labeling_method": "failed",
                    "error": str(e),
                })

            if checkpoint_file and (i + 1) % save_every_n == 0:
                with open(checkpoint_file, "w") as f:
                    json.dump(results, f, indent=2)
                logger.debug(f"Checkpoint saved at {i + 1} images")

        # save final output
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        success_count = sum(1 for r in results if r.get("labeling_method") == "vlm_auto")
        logger.info(
            f"Auto-labeling complete: {success_count}/{len(results)} successful. "
            f"Saved to {output_file}"
        )
        return results
