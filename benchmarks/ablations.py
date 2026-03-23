"""Ablation study configurations for AARS component analysis.

Each :class:`AblationConfig` represents a system variant with one or more
components disabled, allowing us to quantify the marginal contribution
of each subsystem (planner, reflection, fusion, MMR, graph retrieval,
keyword retrieval).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AblationConfig:
    """Immutable specification for an ablation experiment.

    Parameters
    ----------
    name:
        Human-readable identifier (e.g. ``"no_planner"``).
    enable_planner:
        Whether the LLM-based query planner is active.
    enable_reflection:
        Whether the reflection (self-assessment) loop is active.
    enable_fusion:
        Whether multi-source Reciprocal Rank Fusion is used.
    enable_mmr:
        Whether Maximal Marginal Relevance reranking is applied.
    enable_graph:
        Whether the graph (entity-relationship) retriever is available.
    enable_keyword:
        Whether the BM25 keyword retriever is available.
    default_strategy:
        Fallback retrieval strategy when the planner is disabled.
    description:
        Optional free-text description for reporting.
    """

    name: str
    enable_planner: bool = True
    enable_reflection: bool = True
    enable_fusion: bool = True
    enable_mmr: bool = True
    enable_graph: bool = True
    enable_keyword: bool = True
    default_strategy: str = "vector"
    description: str = ""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def to_api_params(self) -> dict[str, Any]:
        """Convert to the parameter dict expected by the AARS API.

        Returns a dictionary that can be merged into the JSON body of a
        ``/query`` request to activate or deactivate the corresponding
        subsystems.
        """
        return {
            "enable_planner": self.enable_planner,
            "enable_reflection": self.enable_reflection,
            "enable_fusion": self.enable_fusion,
            "enable_mmr": self.enable_mmr,
            "enable_graph": self.enable_graph,
            "enable_keyword": self.enable_keyword,
            "default_strategy": self.default_strategy,
        }

    def disabled_components(self) -> list[str]:
        """Return a list of component names that are turned off."""
        mapping = {
            "planner": self.enable_planner,
            "reflection": self.enable_reflection,
            "fusion": self.enable_fusion,
            "mmr": self.enable_mmr,
            "graph": self.enable_graph,
            "keyword": self.enable_keyword,
        }
        return [name for name, enabled in mapping.items() if not enabled]


# ------------------------------------------------------------------
# Pre-defined configurations
# ------------------------------------------------------------------

ABLATION_CONFIGS: list[AblationConfig] = [
    AblationConfig(
        name="full_system",
        description="All AARS components enabled (upper-bound reference).",
    ),
    AblationConfig(
        name="no_planner",
        enable_planner=False,
        default_strategy="hybrid",
        description="Disable the LLM query planner; always use hybrid retrieval.",
    ),
    AblationConfig(
        name="no_reflection",
        enable_reflection=False,
        description="Disable the reflection loop; use first-pass retrieval only.",
    ),
    AblationConfig(
        name="no_fusion",
        enable_fusion=False,
        description="Disable RRF fusion; pass raw retrieval results to generation.",
    ),
    AblationConfig(
        name="no_mmr",
        enable_mmr=False,
        description="Disable MMR diversity reranking after fusion.",
    ),
    AblationConfig(
        name="no_graph",
        enable_graph=False,
        description="Disable graph (entity-relationship) retriever.",
    ),
    AblationConfig(
        name="no_keyword",
        enable_keyword=False,
        default_strategy="vector",
        description="Disable BM25 keyword retriever; rely on dense retrieval only.",
    ),
]


def get_ablation_by_name(name: str) -> AblationConfig | None:
    """Look up a pre-defined ablation config by name.

    Parameters
    ----------
    name:
        Config name to search for (case-sensitive).

    Returns
    -------
    AblationConfig | None
        The matching config, or ``None`` if not found.
    """
    for config in ABLATION_CONFIGS:
        if config.name == name:
            return config
    return None


def build_custom_ablation(
    name: str,
    disabled: list[str] | None = None,
    **overrides: Any,
) -> AblationConfig:
    """Programmatically build an :class:`AblationConfig`.

    Parameters
    ----------
    name:
        Identifier for the config.
    disabled:
        Component names to disable (e.g. ``["planner", "mmr"]``).
    **overrides:
        Additional keyword arguments forwarded to the dataclass constructor.
    """
    flags: dict[str, bool] = {}
    for component in (disabled or []):
        key = f"enable_{component}"
        flags[key] = False
    return AblationConfig(name=name, **flags, **overrides)
