"""Shared 'design-card' paper style for EVERY figure in this paper.

Wraps the base paper style (Nimbus Roman serif, dpi 300) and then forces BOLD figure/axes titles
with a consistent title pad, so every PNG reads like a titled design card (cf. the bold panel titles
in fig05_rolling_schematic). This is the single switch behind items 1 & 13 of the figure-revision goal:
all scripts import `apply` from here instead of importing `paper_style` directly.
"""
import sys

_GNN = "project/manuscript_early/paper_drafting/paper_rewriting_output/gnn_paper_scripts"
if _GNN not in sys.path:
    sys.path.insert(0, _GNN)
from paper_style import apply as _base_apply


def apply():
    """Base publication style + bold titles everywhere (the 'design-card' look)."""
    _base_apply()
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
        "axes.titlepad": 8.0,
        "figure.titleweight": "bold",
    })


def card_title(fig, text, fontsize=16, y=0.995):
    """Bold suptitle helper for full-canvas diagrams that have no axes title."""
    fig.suptitle(text, fontsize=fontsize, fontweight="bold", y=y)
