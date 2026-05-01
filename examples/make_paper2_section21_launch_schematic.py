#!/usr/bin/env python3
"""Generate the Section 2.1 launch-geometry schematic for Paper 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Wedge


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "paper"
    / "paper2_inference_validation"
    / "figures"
    / "forward_model_launch_geometry.png"
)


def configure_matplotlib() -> None:
    """Apply a restrained manuscript style without requiring external LaTeX."""
    matplotlib.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Computer Modern", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path for the generated PNG figure.",
    )
    return parser.parse_args()


def draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "black",
    lw: float = 1.4,
    mutation_scale: float = 14.0,
    alpha: float = 1.0,
    zorder: int = 5,
) -> None:
    """Draw one clean arrow in data coordinates."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        color=color,
        lw=lw,
        mutation_scale=mutation_scale,
        alpha=alpha,
        shrinkA=0.0,
        shrinkB=0.0,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def draw_schematic(output_path: Path) -> None:
    """Draw and save the launch-geometry schematic."""
    configure_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.2, 4.2), constrained_layout=True)
    ax.set_aspect("equal")
    ax.set_xlim(-3.2, 6.6)
    ax.set_ylim(-2.25, 2.25)
    ax.axis("off")

    source = (-2.25, 0.0)
    theta1, theta2 = -23.0, 23.0
    r_outer = 8.0

    cone = Wedge(
        source,
        r_outer,
        theta1,
        theta2,
        facecolor="#e8f1f7",
        edgecolor="#6c8fa8",
        lw=1.2,
        alpha=0.88,
        zorder=1,
    )
    ax.add_patch(cone)

    for angle in np.deg2rad([theta1, theta2]):
        end = (source[0] + r_outer * np.cos(angle), source[1] + r_outer * np.sin(angle))
        ax.plot([source[0], end[0]], [source[1], end[1]], color="#6c8fa8", lw=1.1, zorder=2)

    for radius, alpha in [(1.7, 0.22), (2.8, 0.18), (4.0, 0.15), (5.2, 0.12)]:
        ax.add_patch(
            Arc(
                source,
                2.0 * radius,
                2.0 * radius,
                theta1=theta1,
                theta2=theta2,
                color="0.25",
                lw=0.8,
                alpha=alpha,
                zorder=3,
            )
        )

    source_region = Circle(source, 0.48, facecolor="#f2c166", edgecolor="#8a6124", lw=1.1, zorder=8)
    sonic_radius = Circle(source, 0.78, facecolor="none", edgecolor="#8a6124", lw=1.0, ls="--", zorder=7)
    ax.add_patch(source_region)
    ax.add_patch(sonic_radius)
    ax.text(source[0], source[1] - 0.05, "source\nregion", ha="center", va="center", fontsize=9, zorder=9)

    draw_arrow(ax, source, (source[0] + 0.78, source[1]), color="#8a6124", lw=1.0, mutation_scale=10, zorder=10)
    ax.text(source[0] + 0.42, source[1] + 0.22, r"$r_\star$", fontsize=12, color="#5f4219")

    draw_arrow(ax, (-3.0, 0.36), (source[0] - 0.5, 0.2), color="#bd6b2f", lw=1.2, zorder=9)
    draw_arrow(ax, (-3.0, -0.36), (source[0] - 0.5, -0.2), color="#bd6b2f", lw=1.2, zorder=9)
    ax.text(-3.05, 0.58, r"$\dot{M}_{\rm hot}$", ha="left", va="bottom", fontsize=12, color="#8c4e20")
    ax.text(-3.05, -0.72, r"$\dot{E}_{\rm hot}$", ha="left", va="top", fontsize=12, color="#8c4e20")

    draw_arrow(ax, (source[0] + 0.75, 0.0), (5.95, 0.0), color="#2f6690", lw=2.0, mutation_scale=18, zorder=6)
    draw_arrow(ax, (source[0] + 1.1, 0.45), (5.25, 1.05), color="#2f6690", lw=1.1, alpha=0.65, zorder=5)
    draw_arrow(ax, (source[0] + 1.1, -0.45), (5.25, -1.05), color="#2f6690", lw=1.1, alpha=0.65, zorder=5)
    ax.text(2.0, 0.22, "radial hot wind", ha="center", va="bottom", fontsize=12, color="#1f4d6d")

    omega_arc = Arc(
        source,
        2.55,
        2.55,
        theta1=theta1,
        theta2=theta2,
        color="#476b84",
        lw=1.1,
        zorder=8,
    )
    ax.add_patch(omega_arc)
    ax.text(
        source[0] + 1.25,
        0.34,
        r"$\Omega_{\rm wind}$",
        fontsize=12,
        ha="left",
        va="center",
        color="#365569",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 0.5},
    )

    cloud_data = [
        ((-0.60, 0.16), 0.070, "#4c78a8"),
        ((0.18, -0.18), 0.095, "#59a14f"),
        ((1.05, 0.33), 0.135, "#b07aa1"),
        ((2.02, -0.36), 0.180, "#e15759"),
        ((3.18, 0.18), 0.115, "#f28e2b"),
        ((4.25, -0.10), 0.155, "#76b7b2"),
    ]
    for center, radius, color in cloud_data:
        cloud = Circle(center, radius, facecolor=color, edgecolor="white", lw=0.8, alpha=0.96, zorder=12)
        ax.add_patch(cloud)

    draw_arrow(ax, (2.6, 1.65), (2.03, 0.70), color="0.25", lw=0.9, mutation_scale=10, zorder=11)
    ax.text(
        2.72,
        1.72,
        r"embedded cloud species" + "\n" + r"$M_{{\rm cl},i}$",
        ha="left",
        va="center",
        fontsize=11,
        color="0.18",
    )

    ax.text(
        4.0,
        -1.78,
        r"$\Phi(r)=v_{\rm circ}^2\ln r$",
        ha="center",
        va="center",
        fontsize=12,
        color="0.25",
    )
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    """Generate the manuscript figure."""
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    draw_schematic(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
