"""Simulations accompanying the lung-nematic analysis package.

Two independent models live here, each in its own subpackage because they
define modules with the same names:

    alveolar/    alveolar architecture, epithelial state machine, breathing,
                 coupled mesenchyme, and defect tracking
    fibrofocus/  the standalone fibroblastic-focus model and its bistability
                 analysis of the point of no return
"""

__all__ = ["alveolar", "fibrofocus", "coupled_analysis"]
