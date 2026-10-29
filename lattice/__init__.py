"""Lattice-inspired policy research framework for conversational safety.

The package is intentionally separate from PrefPO's prompt optimizer. It reuses
PrefPO's model, grader, sample, and LLM abstractions while keeping policies as
first-class, versioned rule sets.
"""

from prefpo.lattice.config import (
    AdversarialConfig,
    ConstructionConfig,
    ExperimentConfig,
    ImprovementConfig,
    SimulationConfig,
)
from prefpo.lattice.adapters import ModelAdapter, OfflineModelAdapter, PrefPOModelAdapter
from prefpo.lattice.adversarial import (
    AdversarialBeamSearch,
    AdversarialSearchResult,
)
from prefpo.lattice.datasets import (
    load_conversations,
    load_unlabeled_conversations,
    split_conversations,
)
from prefpo.lattice.evaluation import PolicyEvaluator
from prefpo.lattice.grader import PolicySetGrader
from prefpo.lattice.improvement import ContinuousImprovement
from prefpo.lattice.improvement import ContinuousImprovementResult
from prefpo.lattice.metrics import ClassificationMetrics, evaluate_predictions
from prefpo.lattice.models import (
    Conversation,
    ConversationTurn,
    LabeledConversation,
    PolicyRule,
    RuleSet,
    SafetyLabel,
)
from prefpo.lattice.risk import DualCheckRiskAssessor, RiskAssessment
from prefpo.lattice.rules import RuleConsolidator, RuleGenerator
from prefpo.lattice.optimization import ConstructionResult, PolicyOptimizer
from prefpo.lattice.runner import LatticeExperiment, LatticeResult
from prefpo.lattice.simulation import ConversationSimulator
from prefpo.lattice.tracking import ExperimentTracker

__all__ = [
    "AdversarialConfig",
    "AdversarialBeamSearch",
    "AdversarialSearchResult",
    "ClassificationMetrics",
    "ConstructionConfig",
    "ConstructionResult",
    "ContinuousImprovement",
    "ContinuousImprovementResult",
    "Conversation",
    "ConversationTurn",
    "ExperimentConfig",
    "ExperimentTracker",
    "ImprovementConfig",
    "LabeledConversation",
    "LatticeExperiment",
    "LatticeResult",
    "PolicyEvaluator",
    "PolicySetGrader",
    "ModelAdapter",
    "OfflineModelAdapter",
    "PrefPOModelAdapter",
    "PolicyOptimizer",
    "PolicyRule",
    "RuleSet",
    "RuleConsolidator",
    "RuleGenerator",
    "RiskAssessment",
    "DualCheckRiskAssessor",
    "SafetyLabel",
    "SimulationConfig",
    "ConversationSimulator",
    "evaluate_predictions",
    "load_conversations",
    "load_unlabeled_conversations",
    "split_conversations",
]
