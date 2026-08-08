"""LAP-regularized associative-memory experiments."""

from .experiment import (
    abduct_intervene_predict,
    CausalEnergyMemory,
    ExperimentConfig,
    ModernHopfield,
    RetrievalConfig,
    TrainingConfig,
    generate_chain,
    generate_fork,
    plot_results,
    run_grid,
)

__all__ = [
    "CausalEnergyMemory",
    "abduct_intervene_predict",
    "ExperimentConfig",
    "ModernHopfield",
    "RetrievalConfig",
    "TrainingConfig",
    "generate_chain",
    "generate_fork",
    "plot_results",
    "run_grid",
]
