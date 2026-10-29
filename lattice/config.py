"""Validated experiment configuration for construction and deployment loops."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from prefpo.config import ModelConfig


class StrictConfig(BaseModel):
    """Base config that rejects misspelled experiment parameters."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SimulationConfig(StrictConfig):
    """Multi-turn synthetic conversation settings."""

    turns: int = Field(default=3, ge=1, le=50)
    personas: list[str] = Field(
        default_factory=lambda: [
            "curious_first_time_user",
            "persistent_boundary_tester",
            "context_shifting_user",
        ],
        min_length=1,
    )
    samples_per_example: int = Field(default=1, ge=1, le=100)
    include_assistant_turns: bool = True
    random_seed: int = 42

    @model_validator(mode="after")
    def validate_personas(self) -> SimulationConfig:
        normalized = [persona.strip() for persona in self.personas]
        if any(not persona for persona in normalized):
            raise ValueError("personas must contain non-empty names")
        if len(normalized) != len(set(normalized)):
            raise ValueError("personas must be unique")
        return self


class ConstructionConfig(StrictConfig):
    """Paper-aligned construction stage with explicit tunable bounds."""

    iterations: int = Field(default=10, ge=0, le=1000)
    target_f1: float = Field(default=0.90, ge=0.0, le=1.0)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    minimum_rule_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    consolidate_every_iteration: bool = True
    remove_unused_rules: bool = True
    allow_llm_rule_generation: bool = False


class AdversarialConfig(StrictConfig):
    """Search budget for multi-turn adversarial case expansion."""

    beam_width: int = Field(default=3, ge=1, le=100)
    depth: int = Field(default=10, ge=1, le=100)
    candidates_per_node: int = Field(default=3, ge=1, le=100)
    maximum_nodes: int = Field(default=1000, ge=1)
    retain_blocked_cases: bool = True
    random_seed: int = 42


class ImprovementConfig(StrictConfig):
    """Continuous-improvement guardrails and acceptance criteria."""

    iterations: int = Field(default=3, ge=0, le=1000)
    risk_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    require_non_decreasing_f1: bool = True
    minimum_evidence: int = Field(default=1, ge=1)
    adversarial: AdversarialConfig = Field(default_factory=AdversarialConfig)


class DatasetConfig(StrictConfig):
    """Local data locations and split policy; remote loading is never implicit."""

    train_path: str | None = None
    validation_path: str | None = None
    deployment_path: str | None = None
    format: Literal["jsonl", "json", "csv"] = "jsonl"
    seed: int = 42
    validation_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    require_domain_disjointness: bool = False


class ExperimentConfig(StrictConfig):
    """Top-level, reproducible Lattice experiment specification."""

    name: str = "lattice-research"
    description: str = ""
    seed: int = 42
    model: ModelConfig = Field(default_factory=lambda: ModelConfig(name="openai/gpt-4o-mini"))
    rule_generation_model: ModelConfig | None = None
    simulation_model: ModelConfig | None = None
    construction: ConstructionConfig = Field(default_factory=ConstructionConfig)
    improvement: ImprovementConfig = Field(default_factory=ImprovementConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    output_dir: str = "results/lattice"
    use_deterministic_evaluator: bool = True
    use_offline_adapter: bool = True

    @model_validator(mode="after")
    def validate_components(self) -> ExperimentConfig:
        if not self.name.strip():
            raise ValueError("experiment name must not be empty")
        if not self.output_dir.strip():
            raise ValueError("output_dir must not be empty")
        if self.dataset.require_domain_disjointness:
            if self.dataset.validation_path is None:
                raise ValueError("Domain-disjoint validation requires validation_path")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load and validate YAML; preserve parser and filesystem errors."""
        with Path(path).open(encoding="utf-8") as file:
            raw = yaml.safe_load(file)
        if not isinstance(raw, dict):
            raise ValueError("Experiment YAML must contain a mapping at its root")
        return cls.model_validate(raw)

    def write_yaml(self, path: str | Path) -> None:
        """Write a stable YAML snapshot suitable for experiment tracking."""
        with Path(path).open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                self.model_dump(mode="json"),
                file,
                sort_keys=True,
                allow_unicode=True,
            )
