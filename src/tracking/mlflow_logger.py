"""
MLflow experiment tracking helpers.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from loguru import logger


class MLflowLogger:
    """
    Wraps MLflow tracking calls with sensible defaults and error handling.
    Silently degrades if MLflow is unreachable so training is not interrupted.
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "vit-attribute-classifier",
        run_name: Optional[str] = None,
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._run = None

    def start_run(self, params: Optional[Dict[str, Any]] = None):
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._run = mlflow.start_run(run_name=self.run_name)
            logger.info(f"MLflow run started: {self._run.info.run_id}")
            if params:
                mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"MLflow unavailable: {e}. Continuing without tracking.")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception:
            pass

    def log_params(self, params: Dict[str, Any]):
        try:
            mlflow.log_params(params)
        except Exception:
            pass

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        try:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log artifact {local_path}: {e}")

    def log_model_checkpoint(self, checkpoint_dir: str, step: int):
        """Log a checkpoint directory as an MLflow artifact."""
        self.log_artifact(checkpoint_dir, artifact_path=f"checkpoints/step_{step}")

    def end_run(self, status: str = "FINISHED"):
        try:
            mlflow.end_run(status=status)
        except Exception:
            pass

    @property
    def run_id(self) -> Optional[str]:
        if self._run:
            return self._run.info.run_id
        return None

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FAILED" if exc_type is not None else "FINISHED"
        self.end_run(status=status)
