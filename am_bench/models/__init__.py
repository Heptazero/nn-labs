"""共享模型与默认配置注册表。"""
from typing import Any, Callable
from .classical import ClassicalHopfield
from .dense import PolynomialDAM, ExponentialDAM
from .simplicial import SimplicialR12
from .pshn import PSHN
from .modern import ContinuousModernHopfield
from .base import RetrievalResult

# [注册表] 主实验只通过工厂替换 b/d；a/c/e/f 代码完全复用
MODEL_FACTORIES: dict[str, Callable[[], Any]] = {
    "classical_hebb": ClassicalHopfield,
    "polynomial_dam_d3": lambda: PolynomialDAM(degree=3),
    "exponential_dam": ExponentialDAM,
    "simplicial_r12_t50": lambda: SimplicialR12(0.5, structure_seed=31415),
    "pshn_k8": lambda: PSHN(groups=8),
    "continuous_modern": lambda: ContinuousModernHopfield(beta=1.0),
}
MODEL_FACTORIES