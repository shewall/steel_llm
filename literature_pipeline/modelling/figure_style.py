"""
figure_style.py -- shared, reproducible styling for every dissertation figure
script in this directory. Single source of truth for colours, fonts, panel
labels, target display names, and the multi-format save helper, so every
figure is visually consistent without copy-pasting style constants.

Colour palette is Okabe & Ito (2008), a standard colour-blind-safe palette.
No matplotlib default colour cycle is used anywhere in these scripts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["svg.fonttype"] = "none"  # keep SVG text as real text, not paths
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType (editable) text in PDF, not Type 3

import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito colour-blind-safe palette (named references, used consistently below)
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#999999",
}

# One consistent colour per target, used in every figure that distinguishes targets.
TARGET_COLORS = {
    "UTS_MPa": OKABE_ITO["vermillion"],
    "YS_MPa": OKABE_ITO["bluish_green"],
    "elongation_pct": OKABE_ITO["reddish_purple"],
}

# One consistent colour per model, used in every figure that distinguishes models.
# Deliberately disjoint from TARGET_COLORS so a reader is never asked to decode
# the same colour as meaning two different things across figures.
MODEL_COLORS = {
    "random_forest": OKABE_ITO["blue"],
    "gradient_boosting": OKABE_ITO["orange"],
    "dummy_mean": OKABE_ITO["grey"],
    "ridge": OKABE_ITO["sky_blue"],
    "svr": OKABE_ITO["yellow"],
}

# Consistent target display names/units, used on every axis label.
TARGET_DISPLAY = {
    "UTS_MPa": "UTS (MPa)",
    "YS_MPa": "YS (MPa)",
    "elongation_pct": "Elongation (%)",
}
TARGET_ORDER = ["UTS_MPa", "YS_MPa", "elongation_pct"]

MODEL_DISPLAY = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "dummy_mean": "Dummy (mean)",
    "ridge": "Ridge",
    "svr": "SVR",
}

PANEL_LABEL_STYLE = dict(fontsize=12, fontweight="bold", va="top", ha="left")


def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.05) -> None:
    """Adds a bold panel letter (A/B/C...) in axes-fraction coordinates.
    This is a structural panel label, not a descriptive title -- no figure
    produced by this module uses ax.set_title/fig.suptitle."""
    ax.text(x, y, label, transform=ax.transAxes, **PANEL_LABEL_STYLE)


def support_label_display(support: str) -> str:
    """Maps the shared library's canonical support_label() values to the
    dissertation-facing wording agreed for the figures/captions, WITHOUT
    changing the underlying library function or its tests."""
    return {
        "insufficient_support": "very limited (<5 DOI)",
        "exploratory": "exploratory (5-9 DOI)",
        "supported_descriptive": "supported (>=10 DOI)",
    }.get(support, support)


def save_figure(fig, name: str, out_dir: Path) -> dict:
    """Saves `fig` as a 300dpi PNG under out_dir. Returns {'png': path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", dpi=300)
    paths = {"png": p}
    plt.show()  # inline preview when run from a notebook; a no-op under a non-interactive backend
    plt.close(fig)
    return paths
