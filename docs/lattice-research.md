# Lattice-Inspired Research Framework

This package implements a reproducible research workflow inspired by
[Lattice: Generative Guardrails for Conversational Agents](https://arxiv.org/abs/2601.17481).
It is a research harness, not a production safety product, moderation service,
or claim of reproducing the paper's results.

## What is implemented

The code is under `prefpo.lattice` and keeps policy rules as versioned data:

| Research step | Implementation |
|---|---|
| Rule generation | `RuleGenerator` groups labeled harmful examples by category, emits narrow rules with evidence IDs, and validates model proposals. |
| Rule consolidation | `RuleConsolidator` clusters same-category rules using token Jaccard similarity, optionally delegates consolidation to a model adapter, and preserves lineage. |
| Multi-turn simulation | `ConversationSimulator` samples named personas, extends labeled examples for a configurable number of turns, and generates policy-conditioned assistant responses. |
| Semantic rule evaluation | `PolicyEvaluator` supports an auditable concept/phrase matcher and an asynchronous model-judge path. Every decision includes rule-level matches and rationales. |
| Iterative optimization | `PolicyOptimizer` identifies false negatives, false positives, unused rules, and redundant rules; candidate sets are accepted only when held-out F1 is non-decreasing. |
| Dual-check risk assessment | `DualCheckRiskAssessor` compares the specialized policy with a broader baseline and records missed-threat and false-alarm discrepancies. |
| Adversarial beam search | `AdversarialBeamSearch` expands safe, non-operational test phrasings using a bounded attacker/target loop and classifies successful attacks, blocked attacks, and false alarms. |
| Continuous improvement | `ContinuousImprovement` seeds search from deployment discrepancies, pseudo-labels the resulting evaluation cases, validates candidates on held-out labels, and promotes only accepted policy revisions. |
| Research infrastructure | Local JSON/JSONL/CSV readers, stratified split helpers, precision/recall/F1/confusion counts, bootstrap F1 intervals, JSONL event tracking, policy snapshots, and CSV prediction exports. |

PrefPO integration is intentionally selective. The Lattice model adapter wraps
PrefPO's `call_llm()` and `ModelConfig`; `PolicySetGrader` adapts policies to
PrefPO's existing `Grader` contract. The rule-set optimizer is not a duplicate
of `PromptPool`/PRPO: its state changes are additions, revisions, removals, and
consolidations of typed policies, with a different acceptance objective.

## Paper-specified mechanics vs. reconstructed choices

The paper describes (and this implementation follows at the workflow level):

* Construction from labeled conversations, followed by conversation
  simulation, evaluation, targeted rule edits, and non-regressing F1 promotion.
* Precision, recall, and F1 as the primary classification measures; a user-set
  early-stop threshold.
* Deployment-time comparison of specialized rules and a general safety rule;
  discrepancies seed adversarial case expansion.
* An attacker/target multi-turn beam search, with search leaves labeled as
  successful attacks, blocked attacks, or false alarms.
* Held-out validation before accepting continuous-improvement edits.
* The paper's reported example settings: up to 10 construction iterations,
  three-turn simulation, a target F1 of 0.90, beam width 3, and search depth 10.
  These are defaults in `ConstructionConfig`, `SimulationConfig`, and
  `AdversarialConfig`; experiment YAML can lower budgets for smoke runs.

The paper does not specify enough detail to reproduce every internal operation
from the paper alone. The following are explicit reconstructions in this repo:

* Rule templates, category inference, lexical concept groups, benign-context
  handling, and offline synthetic personas.
* Same-category token-Jaccard consolidation and deterministic tie-breaking.
* The precise priority score used to rank beam nodes, transcript deduplication,
  and the maximum-node safety budget.
* Pseudo-labeling of deployment and adversarial examples from the broad
  baseline, and the precise textual transformations used to broaden/refine
  rules.
* The compact local sample corpus and all reported local results.

The reference paper reports ProsocialDialog results. This repository does not
bundle ProsocialDialog, LlamaGuard, NeMo, or a full 58,000-example corpus; the
included `research_data/*.jsonl` records are authored synthetic examples for
offline tests and interface demonstrations. To evaluate an external corpus,
provide a local JSON/JSONL/CSV file with `id`, `domain`, `category`, `label`,
and `turns`/`messages` fields. `load_conversations()` intentionally does not
download data or transmit private conversations.

## Offline construction run

From the repository root:

```powershell
python -m prefpo.lattice.cli `
  --config research_configs/construction.yaml
```

The CLI is deterministic and offline by default. It writes one unique run
directory containing:

* `config.json` — validated configuration snapshot.
* `events.jsonl` — sequenced dataset, construction, metric, and completion events.
* `policies/` — fingerprinted policy snapshots with provenance.
* `construction-validation-metrics.json` — confusion counts and overall/sliced
  precision, recall, F1, accuracy, and specificity.
* `summary.json` — final selected policy fingerprint and result summary.

The continuous-improvement sample config also reads
`research_data/deployment.jsonl`. Its deliberately small beam depth is a fast
smoke-test override; the research defaults remain 3 by 10. Run commands against
real model providers only after explicitly setting `use_offline_adapter: false`
and configuring the required API credentials in the environment. That mode
uses the PrefPO `call_llm()` provider stack.

## Python API

```python
import asyncio
from prefpo.lattice import ExperimentConfig, LatticeExperiment, load_conversations

config = ExperimentConfig.from_yaml("research_configs/construction.yaml")
records = load_conversations("research_data/construction.jsonl")
result = asyncio.run(LatticeExperiment(config).run(train=records))
print(result.construction.best_report.overall.to_dict())
print(result.run_directory)
```

For a fully explicit split, pass both `train` and `validation`. If validation is
omitted, the experiment runner makes a seeded, approximate-stratified split.
`validate_domain_disjointness()` can enforce cross-domain evaluation when
different domains are provided. Do not select or tune a policy on the test set;
the runner only reports test metrics after construction and improvement.

An alternate semantic judge can be selected with
`PolicyEvaluator(adapter=..., semantic_mode="model")`. Its asynchronous
`evaluate_async()` path is used by construction, simulation, risk assessment,
beam search, and the PrefPO grader adapter. The lexical evaluator is the
default because it is deterministic, auditable, and usable without API access;
it is a transparent baseline, not a substitute for human-reviewed semantic
evaluation.

## Evaluation and tracking details

The positive class is `harmful` (`1`); `benign` is `0`. Empty denominators yield
zero, making degenerate results explicit. `ClassificationMetrics` includes
TP/FP/FN/TN, precision, recall, F1, accuracy, specificity, and support.
`evaluate_predictions()` can additionally report domain/category slices and a
seeded percentile bootstrap interval. These intervals describe variation in
resampled evaluation rows only; they do not capture annotator or model
uncertainty.

`ExperimentTracker` creates a unique run directory instead of overwriting
earlier results. Event writes are append-only and fsynced; policy and summary
snapshots are written through a temporary file followed by atomic replacement.
Concurrent append calls are synchronized within a process. This is a local
research tracker, not a distributed experiment database.

## Extension points

* Implement `ModelAdapter.run(task, payload)` to connect a research model,
  simulator, or test harness. `PrefPOModelAdapter` is the litellm-backed option;
  `OfflineModelAdapter` provides deterministic, non-network operation.
* Supply `PolicyEvaluator` to `PolicyOptimizer` to change evaluation behavior.
  Custom semantic judges must validate their booleans and confidence values.
* Implement different rule generation or consolidation behavior behind the
  adapter operations while keeping rule IDs and `evidence_ids` traceable.
* Use `PolicySetGrader` to evaluate rule sets through existing PrefPO grading
  code. `as_prefpo_sample()` and `from_prefpo_samples()` bridge labeled
  conversations to PrefPO `Sample` values.

## Testing

The Lattice tests use synthetic, local data and fake/offline adapters. They
should never make live API calls. Existing live PrefPO tests remain opt-in under
the repository's pytest marker configuration. A test failure should be
reported with its original traceback rather than hidden by the experiment
runner.
