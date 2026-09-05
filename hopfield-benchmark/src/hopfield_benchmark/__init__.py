"""Task-centered Hopfield benchmark components."""

from .components import (
    MemorySet,
    RetrievalResult,
    TrialSpec,
    a1_make_independent_binary,
    c1_make_hamming_cue,
    e1_exact_recall,
    e2_overlap,
    e3_attractor_class,
    e4_top1_memory,
)
from .models import ClassicalHopfield, ModelAdapter, PolynomialDAM
from .plotting import (
    f1_plot_recall_vs_corruption,
    f2_plot_recall_vs_load,
    f3_plot_capacity_scaling,
    f4_plot_quality_cost,
    f5_plot_error_dynamics,
)
from .runner import (
    BenchmarkConfig,
    capacity_summary,
    run_paired_benchmark,
    validate_paired_results,
)

__all__ = [
    "BenchmarkConfig",
    "ClassicalHopfield",
    "MemorySet",
    "ModelAdapter",
    "PolynomialDAM",
    "RetrievalResult",
    "TrialSpec",
    "a1_make_independent_binary",
    "c1_make_hamming_cue",
    "capacity_summary",
    "e1_exact_recall",
    "e2_overlap",
    "e3_attractor_class",
    "e4_top1_memory",
    "f1_plot_recall_vs_corruption",
    "f2_plot_recall_vs_load",
    "f3_plot_capacity_scaling",
    "f4_plot_quality_cost",
    "f5_plot_error_dynamics",
    "run_paired_benchmark",
    "validate_paired_results",
]
