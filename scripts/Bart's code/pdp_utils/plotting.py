# -*- coding: utf-8 -*-
"""
Plotly and Matplotlib drawing helpers for lane markings, annotations,
intersection roads, and Frenet coordinate axes.

All functions are pure — they accept explicit parameters and do not access
globals or ``st.session_state``.
"""

from typing import Any, Tuple, Optional

import numpy as np
import matplotlib.axes
import plotly.graph_objects as go  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

def add_lane_polylines_plotly(
    fig: go.Figure,
    lane_polylines: dict[str, Any],
    lane_color: str,
    edge_color: str,
    dashed_color: str,
) -> go.Figure:
    """Add lane boundary/divider polylines to a Plotly figure."""
    boundaries = lane_polylines.get("boundaries", [])
    center_lines: list[np.ndarray] = lane_polylines.get("center_lines", [])

    if len(boundaries) >= 2:
        left_edge = boundaries[0]
        right_edge = boundaries[-1]
        polygon_x = np.concatenate([left_edge[:, 0], right_edge[::-1, 0], [left_edge[0, 0]]])
        polygon_y = np.concatenate([left_edge[:, 1], right_edge[::-1, 1], [left_edge[0, 1]]])

        fig.add_trace(
            go.Scatter(
                x=polygon_x,
                y=polygon_y,
                fill="toself",
                fillcolor=lane_color,
                line=dict(color="rgba(0, 0, 0, 0)", width=0),
                hoverinfo="skip",
                showlegend=False,
                name="Road surface",
            )
        )

        for edge in (left_edge, right_edge):
            fig.add_trace(  # type: ignore[call-arg]
                go.Scatter(
                    x=edge[:, 0],
                    y=edge[:, 1],
                    mode="lines",
                    line=dict(color=edge_color, width=3),
                    hoverinfo="skip",
                    showlegend=False,
                    name="Road edge",
                )
            )

    for dashed_line in center_lines:
        fig.add_trace(  # type: ignore[call-arg]
            go.Scatter(
                x=dashed_line[:, 0],
                y=dashed_line[:, 1],
                mode="lines",
                line=dict(color=dashed_color, width=2, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
                name="Lane divider",
            )
        )

    return fig


def add_intersection_lanes_plotly(
    fig: go.Figure,
    config: dict[str, Any],
    lane_color: str,
    line_color: str,
    dashed_color: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> go.Figure:
    """Add intersection (crossing roads) lane markings to a Plotly figure."""
    lanes_h = config["lanes_horizontal"]
    lanes_v = config["lanes_vertical"]
    lane_width = config["lane_width"]
    cx, cy = config["center"]

    off_h = float(config.get("offset_horizontal", 0.0))
    off_v = float(config.get("offset_vertical", 0.0))
    cy += off_h
    cx += off_v

    h_x_min = xlim[0] if xlim else config.get("horizontal_range", (cx - 100, cx + 100))[0]
    h_x_max = xlim[1] if xlim else config.get("horizontal_range", (cx - 100, cx + 100))[1]
    v_y_min = ylim[0] if ylim else config.get("vertical_range", (cy - 100, cy + 100))[0]
    v_y_max = ylim[1] if ylim else config.get("vertical_range", (cy - 100, cy + 100))[1]

    h_width = lanes_h * lane_width
    h_y_bottom = cy - h_width / 2
    h_y_top = cy + h_width / 2

    v_width = lanes_v * lane_width
    v_x_left = cx - v_width / 2
    v_x_right = cx + v_width / 2

    # Road areas
    fig.add_shape(  # type: ignore[call-arg]
        type="rect", x0=h_x_min, y0=h_y_bottom, x1=h_x_max, y1=h_y_top,
        fillcolor=lane_color, line=dict(color="rgba(0,0,0,0)", width=0), layer="below",
    )
    fig.add_shape(  # type: ignore[call-arg]
        type="rect", x0=v_x_left, y0=v_y_min, x1=v_x_right, y1=v_y_max,
        fillcolor=lane_color, line=dict(color="rgba(0,0,0,0)", width=0), layer="below",
    )

    # Horizontal lane dividers
    for i in range(1, lanes_h):
        y_line = h_y_bottom + i * lane_width
        for seg_x in ([h_x_min, v_x_left], [v_x_right, h_x_max]):
            fig.add_trace(go.Scatter(  # type: ignore[call-arg]
                x=seg_x, y=[y_line, y_line], mode="lines",
                line=dict(color=dashed_color, width=1.5, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))

    # Vertical lane dividers
    for i in range(1, lanes_v):
        x_line = v_x_left + i * lane_width
        for seg_y in ([v_y_min, h_y_bottom], [h_y_top, v_y_max]):
            fig.add_trace(go.Scatter(  # type: ignore[call-arg]
                x=[x_line, x_line], y=seg_y, mode="lines",
                line=dict(color=dashed_color, width=1.5, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))

    # Horizontal edge lines
    for y_edge in (h_y_bottom, h_y_top):
        for seg_x in ([h_x_min, v_x_left], [v_x_right, h_x_max]):
            fig.add_trace(go.Scatter(  # type: ignore[call-arg]
                x=seg_x, y=[y_edge, y_edge], mode="lines",
                line=dict(color=line_color, width=2),
                showlegend=False, hoverinfo="skip",
            ))

    # Vertical edge lines
    for x_edge in (v_x_left, v_x_right):
        for seg_y in ([v_y_min, h_y_bottom], [h_y_top, v_y_max]):
            fig.add_trace(go.Scatter(  # type: ignore[call-arg]
                x=[x_edge, x_edge], y=seg_y, mode="lines",
                line=dict(color=line_color, width=2),
                showlegend=False, hoverinfo="skip",
            ))

    return fig


# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------

def draw_frenet_axes(
    ax: matplotlib.axes.Axes,
    centerline: np.ndarray,
    num_arrows: int = 5,
) -> None:
    """Draw subtle Frenet coordinate axes (tangent=d1, normal=d2) along the centerline."""
    if len(centerline) < 2:
        return

    tangents = np.zeros_like(centerline)
    tangents[1:-1] = centerline[2:] - centerline[:-2]
    tangents[0] = centerline[1] - centerline[0]
    tangents[-1] = centerline[-1] - centerline[-2]

    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    tangents = tangents / norms

    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    indices = np.linspace(0, len(centerline) - 1, num_arrows + 2, dtype=int)[1:-1]

    arrow_length = 6.0
    arrow_alpha = 0.7

    for idx in indices:  # type: ignore[misc]
        pos = centerline[idx]
        T = tangents[idx]
        N = normals[idx]

        ax.arrow(  # type: ignore[call-arg]
            pos[0], pos[1], T[0] * arrow_length, T[1] * arrow_length,
            head_width=1.2, head_length=0.8, fc="#2166ac", ec="#2166ac",
            alpha=arrow_alpha, zorder=3, linewidth=0.8,
        )
        label_pos = pos + T * (arrow_length + 2.0)
        ax.text(  # type: ignore[call-overload]
            label_pos[0], label_pos[1], "d1", fontsize=7, color="#2166ac",
            alpha=arrow_alpha, ha="center", va="center", zorder=3, fontweight="bold",
        )

        ax.arrow(  # type: ignore[call-arg]
            pos[0], pos[1], N[0] * arrow_length, N[1] * arrow_length,
            head_width=1.2, head_length=0.8, fc="#b2182b", ec="#b2182b",
            alpha=arrow_alpha, zorder=3, linewidth=0.8,
        )
        label_pos = pos + N * (arrow_length + 2.0)
        ax.text(  # type: ignore[call-overload]
            label_pos[0], label_pos[1], "d2", fontsize=7, color="#b2182b",
            alpha=arrow_alpha, ha="center", va="center", zorder=3, fontweight="bold",
        )


def draw_intersection_lanes_matplotlib(
    ax: matplotlib.axes.Axes,
    config: dict[str, Any],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    """Draw intersection lane markings (crossing roads) on a Matplotlib axes."""
    lane_width = float(config.get("lane_width", 3.0))
    lanes_h = int(config.get("lanes_horizontal", 3))
    lanes_v = int(config.get("lanes_vertical", 3))
    center_x, center_y = config.get("center", (0.0, 0.0))

    off_h = float(config.get("offset_horizontal", 0.0))
    off_v = float(config.get("offset_vertical", 0.0))
    center_y += off_h
    center_x += off_v

    h_x_min = xlim[0]
    h_x_max = xlim[1]
    v_y_min = ylim[0]
    v_y_max = ylim[1]

    road_color = "none"
    edge_line_color = "black"
    center_line_color = "black"
    lane_line_width = 0.8

    h_half = lane_width * lanes_h / 2.0
    v_half = lane_width * lanes_v / 2.0

    # Road fills
    h_poly = np.array([
        [h_x_min, center_y - h_half], [h_x_max, center_y - h_half],
        [h_x_max, center_y + h_half], [h_x_min, center_y + h_half],
    ])
    ax.fill(h_poly[:, 0], h_poly[:, 1], facecolor=road_color, edgecolor="none", zorder=0)  # type: ignore[call-overload]

    v_poly = np.array([
        [center_x - v_half, v_y_min], [center_x + v_half, v_y_min],
        [center_x + v_half, v_y_max], [center_x - v_half, v_y_max],
    ])
    ax.fill(v_poly[:, 0], v_poly[:, 1], facecolor=road_color, edgecolor="none", zorder=0)  # type: ignore[call-overload]

    # Horizontal edges (skip intersection box)
    for y_edge in (center_y - h_half, center_y + h_half):
        ax.plot([h_x_min, center_x - v_half], [y_edge, y_edge], color=edge_line_color, linewidth=lane_line_width, alpha=1.0, zorder=1)  # type: ignore[call-overload]
        ax.plot([center_x + v_half, h_x_max], [y_edge, y_edge], color=edge_line_color, linewidth=lane_line_width, alpha=1.0, zorder=1)  # type: ignore[call-overload]

    # Vertical edges (skip intersection box)
    for x_edge in (center_x - v_half, center_x + v_half):
        ax.plot([x_edge, x_edge], [v_y_min, center_y - h_half], color=edge_line_color, linewidth=lane_line_width, alpha=1.0, zorder=1)  # type: ignore[call-overload]
        ax.plot([x_edge, x_edge], [center_y + h_half, v_y_max], color=edge_line_color, linewidth=lane_line_width, alpha=1.0, zorder=1)  # type: ignore[call-overload]

    # Horizontal dashed dividers
    for i in range(1, lanes_h):
        y_val = center_y - h_half + i * lane_width
        for seg_x in ([h_x_min, center_x - v_half], [center_x + v_half, h_x_max]):
            ax.plot(seg_x, [y_val, y_val], color=center_line_color, linewidth=lane_line_width,
                    linestyle="--", dashes=(10, 10), alpha=1.0, zorder=1)

    # Vertical dashed dividers
    for i in range(1, lanes_v):
        x_val = center_x - v_half + i * lane_width
        for seg_y in ([v_y_min, center_y - h_half], [center_y + h_half, v_y_max]):
            ax.plot([x_val, x_val], seg_y, color=center_line_color, linewidth=lane_line_width,
                    linestyle="--", dashes=(10, 10), alpha=1.0, zorder=1)


def annotate_points(
    ax: matplotlib.axes.Axes,
    pts: np.ndarray,
    ts: np.ndarray,
    label_prefix: str,
    color: str,
    label_fs: int = 9,
) -> None:
    """Draw scatter points plus label subscripts (e.g. ``k_0``, ``l_1``)."""
    offsets = [(3, 3), (3, -8), (-8, 3)]
    for i, ((x, y), tval) in enumerate(zip(pts, ts)):  # type: ignore[misc]
        ax.scatter([x], [y], s=25, zorder=10, color=color, marker="o")
        off = offsets[i % len(offsets)]
        try:
            tnum = float(tval)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            tnum = float(np.array(tval, dtype=float))
        lbl = str(int(tnum)) if tnum.is_integer() else f"{tnum:g}"
        label_text = f"$\\mathit{{{label_prefix}}}_{{{lbl}}}$"
        ax.annotate(  # type: ignore
            label_text,
            xy=(x, y),
            xytext=off,
            textcoords="offset points",
            fontsize=label_fs,
            color=color,
            ha="left" if off[0] >= 0 else "right",
            va="bottom" if off[1] >= 0 else "top",
        )
