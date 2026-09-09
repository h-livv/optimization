"""Visualize simplex iterates in a fixed decision-variable coordinate system.

Each recorded basis and x_B is turned back into a full standard-form vector:

    x[basic]     = x_B
    x[non-basic] = 0

The plot then shows only the three decision coordinates (x1, x2, x3).
Those axes stay the same for every iteration. This is the usual geometric
picture of walking around the feasible polytope; it is not a plot of the
changing basic-variable coordinate system.
"""

import warnings
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


DISPLAY_ATOL = 1e-10


def _display_coords(values):
    coords = np.asarray(values, dtype=float).reshape(-1).copy()
    coords[np.isclose(coords, 0.0, atol=DISPLAY_ATOL)] = 0.0
    return coords


def _fmt_num(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    value = float(np.asarray(value).reshape(-1)[0])
    if abs(value) < DISPLAY_ATOL:
        return "0"
    nearest = round(value)
    if abs(value - nearest) < 1e-8:
        return str(int(nearest))
    return f"{value:.4g}"


def _is_unit_column(col):
    col = np.asarray(col, dtype=float).reshape(-1)
    return (
        np.sum(np.isclose(col, 1.0, atol=1e-8)) == 1
        and np.sum(np.isclose(col, 0.0, atol=1e-8)) == col.size - 1
    )


def _decision_index(A):
    """Columns that are not identity slacks; fall back to the first three."""
    n = A.shape[1]
    slacks = [j for j in range(n) if _is_unit_column(A[:, j])]
    remaining = [j for j in range(n) if j not in slacks]
    if len(remaining) == 3:
        return remaining
    if n >= 3:
        return list(range(3))
    raise ValueError("Need at least 3 variables to plot in (x1, x2, x3)-space.")


def _full_x(record, n_vars):
    # x_B occupies the current basic positions; every non-basic variable is 0.
    x = np.zeros(n_vars, dtype=float)
    x[list(record["basic_index"])] = np.asarray(record["x_b"], dtype=float).reshape(-1)
    return x


def _decision_point(record, n_vars, decision_index):
    return _full_x(record, n_vars)[list(decision_index)]


def _reconstruct_b(history, A):
    rec = history[0]
    return A[:, rec["basic_index"]] @ np.asarray(rec["x_b"], dtype=float)


def _inequality_system(A, b, decision_index):
    """Build G x_d <= h from A x = b for the plotted decision variables.

    Split A into decision columns and the remaining (slack) columns. Then

        A_s x_s = b - A_d x_d,  x_s >= 0

    becomes

        A_s^{-1} A_d x_d <= A_s^{-1} b,  x_d >= 0

    Identity-column detection is not used here: a decision variable such as
    x3 can itself be a unit column and is still part of x_d.
    """
    m, n = A.shape
    slacks = [j for j in range(n) if j not in set(decision_index)]
    if len(slacks) != m:
        return None, None

    A_d = A[:, list(decision_index)]
    A_s = A[:, slacks]
    try:
        A_s_inv = np.linalg.inv(A_s)
    except np.linalg.LinAlgError:
        return None, None

    G = np.vstack([A_s_inv @ A_d, -np.eye(len(decision_index))])
    h = np.concatenate([A_s_inv @ b, np.zeros(len(decision_index))])
    return G, h


def _unique_points(points, tol=1e-8):
    unique = []
    for p in points:
        p = np.asarray(p, dtype=float).reshape(-1)
        if not any(np.allclose(p, q, atol=tol) for q in unique):
            unique.append(p)
    return unique


def _polytope_vertices(G, h):
    dim = G.shape[1]
    vertices = []
    for rows in combinations(range(len(h)), dim):
        G3 = G[list(rows), :]
        if abs(np.linalg.det(G3)) < 1e-10:
            continue
        x = np.linalg.solve(G3, h[list(rows)])
        if np.all(G @ x <= h + 1e-7):
            vertices.append(x)
    return _unique_points(vertices)


def _order_polygon(points):
    pts = np.asarray(points, dtype=float)
    rel = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(rel)
    angles = np.arctan2(rel @ vh[1], rel @ vh[0])
    return pts[np.argsort(angles)]


def _polytope_faces(vertices, G, h):
    faces = []
    verts = np.asarray(vertices, dtype=float)
    for i in range(len(h)):
        on_plane = [v for v in verts if np.isclose(G[i] @ v, h[i], atol=1e-6)]
        if len(on_plane) >= 3:
            faces.append(_order_polygon(on_plane))
    return faces


def _polytope_edges(vertices, G, h):
    verts = [np.asarray(v, dtype=float) for v in vertices]
    tight = [
        {i for i in range(len(h)) if np.isclose(G[i] @ v, h[i], atol=1e-6)}
        for v in verts
    ]
    edges = []
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            if len(tight[i] & tight[j]) >= G.shape[1] - 1:
                edges.append((verts[i], verts[j]))
    return edges


def _axis_bounds(points):
    stacked = np.vstack([_display_coords(p) for p in points])
    lo = np.minimum(np.min(stacked, axis=0), 0.0)
    hi = np.maximum(np.max(stacked, axis=0), 0.0)
    lo[np.isclose(lo, 0.0, atol=DISPLAY_ATOL)] = 0.0
    span = hi - lo
    pad = np.where(span > 1e-8, 0.18 * span, 1.0)
    return lo, hi + pad


def _plot_arrow(ax, start, end, lo, hi, color="C1", linewidth=2.5, alpha=1.0):
    """Shaft plus a small 3D arrowhead sized in visual (axis-fraction) space.

    Matplotlib's 3D quiver uses data coordinates for the head, which becomes
    a stray segment when the axis scales differ. Scaling by (hi - lo) keeps
    the head a compact V at the tip.
    """
    start = _display_coords(start)
    end = _display_coords(end)
    span = np.maximum(np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float), 1e-12)
    visual = (end - start) / span
    length = np.linalg.norm(visual)
    if length < 1e-12:
        return

    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
    )

    direction = visual / length
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    side = np.cross(direction, helper)
    side = side / (np.linalg.norm(side) + 1e-12)

    head = min(0.05, 0.4 * length)
    left = end + (-direction * head + side * 0.35 * head) * span
    right = end + (-direction * head - side * 0.35 * head) * span
    ax.plot(
        [end[0], left[0]], [end[1], left[1]], [end[2], left[2]],
        color=color, linewidth=linewidth, alpha=alpha,
    )
    ax.plot(
        [end[0], right[0]], [end[1], right[1]], [end[2], right[2]],
        color=color, linewidth=linewidth, alpha=alpha,
    )


def _reduced_costs(record, A, c):
    basic = list(record["basic_index"])
    non_basic = [i for i in range(A.shape[1]) if i not in basic]
    B_inv = np.linalg.inv(A[:, basic])
    c_B = c[basic]
    c_N = c[non_basic]
    N = A[:, non_basic]
    return non_basic, c_N - c_B.T @ B_inv @ N


def _ratio_rows(record, variable_names):
    if record["d"] is None:
        return []
    d = np.asarray(record["d"], dtype=float).reshape(-1)
    x_b = np.asarray(record["x_b"], dtype=float).reshape(-1)
    basic = list(record["basic_index"])
    rows = []
    for i, var in enumerate(basic):
        if d[i] > 0:
            rows.append((variable_names[var], x_b[i] / d[i]))
        else:
            rows.append((variable_names[var], None))
    return rows


def _validate_path(history, n_vars, decision_index):
    for i, record in enumerate(history[:-1]):
        if record["entering_variable_index"] is None:
            continue
        nxt = history[i + 1]
        x_next = _full_x(nxt, n_vars)
        leaving = record["leaving_variable_index"]
        entering = record["entering_variable_index"]
        t_max = float(record["selected_ratio"])
        if not np.isclose(x_next[leaving], 0.0, atol=1e-7):
            warnings.warn(
                f"Iteration {record['iteration']}: leaving variable "
                f"{leaving} is {x_next[leaving]} at the next BFS (expected 0).",
                RuntimeWarning,
                stacklevel=2,
            )
        if not np.isclose(x_next[entering], t_max, atol=1e-7):
            warnings.warn(
                f"Iteration {record['iteration']}: entering variable "
                f"{entering} is {x_next[entering]} at the next BFS "
                f"(expected t_max = {t_max}).",
                RuntimeWarning,
                stacklevel=2,
            )
        here = _decision_point(record, n_vars, decision_index)
        there = _decision_point(nxt, n_vars, decision_index)
        if np.allclose(here, there, atol=1e-10):
            warnings.warn(
                f"Iteration {record['iteration']}: consecutive BFS points "
                "coincide in decision space.",
                RuntimeWarning,
                stacklevel=2,
            )


def _info_text(record, A, c, variable_names, decision_index, n_vars):
    x = _display_coords(_full_x(record, n_vars))
    labels = [variable_names[i] for i in record["basic_index"]]
    x_b = _display_coords(record["x_b"])
    status = "Optimal" if record["entering_variable_index"] is None else "Pivoting"

    lines = [
        f"Iteration: {record['iteration']}",
        f"Status: {status}",
        "",
        "Decision variables:",
        *[
            f"  {variable_names[j]} = {_fmt_num(x[j])}"
            for j in decision_index
        ],
        "",
        f"Basis: {labels}",
        "x_B:",
        *[f"  {labels[i]} = {_fmt_num(x_b[i])}" for i in range(len(labels))],
        f"Objective: {_fmt_num(record['objective_value'])}",
    ]

    if c is not None:
        non_basic, reduced = _reduced_costs(record, A, c)
        lines.extend(["", "Reduced costs:"])
        for idx, cost in zip(non_basic, np.asarray(reduced).reshape(-1)):
            lines.append(f"  {variable_names[idx]}: {_fmt_num(cost)}")

    if record["entering_variable_index"] is not None:
        lines.extend([
            "",
            f"Entering: {variable_names[record['entering_variable_index']]}",
            f"Leaving: {variable_names[record['leaving_variable_index']]}",
            "",
            "Ratio test (x_B[i] / d[i] for d[i] > 0):",
        ])
        for name, ratio in _ratio_rows(record, variable_names):
            mark = ""
            if (
                ratio is not None
                and np.isclose(ratio, record["selected_ratio"], atol=1e-8)
                and name == variable_names[record["leaving_variable_index"]]
            ):
                mark = "  ← min"
            lines.append(f"  {name}: {_fmt_num(ratio)}{mark}")
        lines.append(f"t_max = {_fmt_num(record['selected_ratio'])}")
    else:
        lines.extend(["", "No improving entering variable.", "Optimal"])

    return "\n".join(lines)


def _draw_polytope(ax, faces, edges):
    if faces:
        collection = Poly3DCollection(
            faces,
            facecolors=(0.55, 0.70, 0.90, 0.14),
            edgecolors=(0.20, 0.35, 0.55, 0.85),
            linewidths=1.2,
        )
        collection.set_zsort("min")
        ax.add_collection3d(collection)
    for p, q in edges:
        p = _display_coords(p)
        q = _display_coords(q)
        ax.plot(
            [p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
            color=(0.25, 0.40, 0.60),
            linewidth=1.6,
            zorder=2,
        )


def _draw_iteration(
    ax,
    ax_info,
    history,
    frame_index,
    A,
    c,
    variable_names,
    decision_index,
    axis_labels,
    lo,
    hi,
    faces,
    edges,
    path,
):
    if getattr(ax, "_simplex_view_inited", False):
        elev, azim = ax.elev, ax.azim
    else:
        # Elevated view into the first octant so x1, x2, and x3 are all readable.
        elev, azim = 32, 45
        ax._simplex_view_inited = True

    ax.cla()
    ax_info.cla()
    ax_info.axis("off")

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    # Keep data limits on each axis, but do not scale the 3D box by those
    # ranges. Equal data-aspect turns a problem with b = [5, 25, 125] into a
    # needle. A cube is readable whenever the axis spans differ a lot;
    # similar spans still get a data-proportional box.
    span = np.maximum(hi - lo, 1e-12)
    aspect = span if span.max() / span.min() < 4 else (1.0, 1.0, 1.0)
    try:
        ax.set_box_aspect(aspect)
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)

    # Axes are the original decision variables and do not change after a pivot.
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])

    record = history[frame_index]
    ax.set_title(
        f"Iteration {record['iteration']}   "
        f"Basis = [{', '.join(variable_names[i] for i in record['basic_index'])}]"
    )

    _draw_polytope(ax, faces, edges)

    ax.plot([lo[0], hi[0]], [0, 0], [0, 0], color="0.8", linewidth=0.8)
    ax.plot([0, 0], [lo[1], hi[1]], [0, 0], color="0.8", linewidth=0.8)
    ax.plot([0, 0], [0, 0], [lo[2], hi[2]], color="0.8", linewidth=0.8)
    ax.scatter(0, 0, 0, color="k", s=15)

    # Keep every earlier pivot arrow so the path reads as a sequence of steps.
    for i in range(frame_index):
        prev = _display_coords(path[i])
        nxt = _display_coords(path[i + 1])
        _plot_arrow(ax, prev, nxt, lo, hi, color="#c45c26", linewidth=3.0, alpha=1.0)

    start = _display_coords(path[frame_index])

    if frame_index + 1 < len(path) and record["entering_variable_index"] is not None:
        nxt = _display_coords(path[frame_index + 1])
        _plot_arrow(ax, start, nxt, lo, hi, color="C1", linewidth=2.8)

    ax.scatter(
        start[0], start[1], start[2],
        s=320, facecolors="none", edgecolors="C0", linewidths=2.4,
        alpha=0.45, zorder=9,
    )
    ax.scatter(
        start[0], start[1], start[2],
        color="C0", s=170, edgecolors="k", linewidths=1.4,
        zorder=10, label="current BFS",
    )
    ax.text(
        start[0], start[1], start[2],
        "  current BFS",
        color="C0", fontsize=10, fontweight="bold", zorder=12,
    )

    ax.legend(
        loc="upper left",
        handles=[
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                markerfacecolor="C0",
                markeredgecolor="k",
                markersize=10,
                label="current BFS",
            ),
            Line2D(
                [0], [0],
                color="#c45c26",
                linewidth=3.0,
                label="pivot direction",
            ),
        ],
    )
    ax_info.text(
        0.0, 1.0,
        _info_text(record, A, c, variable_names, decision_index, A.shape[1]),
        va="top", ha="left", family="monospace", fontsize=9,
        transform=ax_info.transAxes,
    )


def visualize_simplex(
    history,
    A,
    variable_names=None,
    b=None,
    c=None,
    animate=True,
    interval=2500,
    show=True,
):
    """Plot simplex BFS points in a fixed 3D decision-variable frame."""
    if not history:
        raise ValueError("history is empty")

    A = np.asarray(A, dtype=float)
    if b is not None:
        b = np.asarray(b, dtype=float)
    else:
        b = _reconstruct_b(history, A)
    if c is not None:
        c = np.asarray(c, dtype=float)

    if variable_names is None:
        variable_names = [f"x{i + 1}" for i in range(A.shape[1])]
    else:
        variable_names = list(variable_names)

    decision_index = _decision_index(A)
    axis_labels = [variable_names[i] for i in decision_index]
    n_vars = A.shape[1]

    path = [
        _display_coords(_decision_point(rec, n_vars, decision_index))
        for rec in history
    ]
    _validate_path(history, n_vars, decision_index)

    faces, edges = [], []
    G, h = _inequality_system(A, b, decision_index)
    vertices = []
    if G is not None:
        vertices = _polytope_vertices(G, h)
        if vertices:
            faces = _polytope_faces(vertices, G, h)
            edges = _polytope_edges(vertices, G, h)

    bound_points = list(path) + [np.zeros(3)] + list(vertices)
    lo, hi = _axis_bounds(bound_points)

    fig = plt.figure("Simplex", figsize=(12, 7))
    try:
        fig.canvas.manager.set_window_title("Simplex")
    except Exception:
        pass
    grid = GridSpec(1, 2, width_ratios=[1.55, 1], wspace=0.05, bottom=0.10, top=0.90)
    ax = fig.add_subplot(grid[0], projection="3d")
    ax_info = fig.add_subplot(grid[1])
    fig.suptitle(
        f"Simplex BFS in fixed ({', '.join(axis_labels)})-space",
        fontsize=12,
    )

    n_frames = len(history)
    state = {
        "index": 0,
        "playing": bool(animate and n_frames > 1),
    }

    def draw_current():
        _draw_iteration(
            ax, ax_info, history, state["index"],
            A, c, variable_names, decision_index, axis_labels,
            lo, hi, faces, edges, path,
        )
        play_button.label.set_text("Pause" if state["playing"] else "Play")
        fig.canvas.draw_idle()

    def set_playing(playing):
        state["playing"] = playing
        if playing and state["index"] >= n_frames - 1:
            state["index"] = 0
        draw_current()

    def on_timer():
        if not state["playing"]:
            return
        if state["index"] < n_frames - 1:
            state["index"] += 1
            draw_current()
        else:
            state["playing"] = False
            draw_current()

    def on_key(event):
        if event.key == " ":
            set_playing(not state["playing"])
        elif event.key in ("right", "n"):
            state["playing"] = False
            state["index"] = min(state["index"] + 1, n_frames - 1)
            draw_current()
        elif event.key in ("left", "p"):
            state["playing"] = False
            state["index"] = max(state["index"] - 1, 0)
            draw_current()

    play_ax = fig.add_axes([0.08, 0.02, 0.12, 0.045])
    play_button = Button(play_ax, "Pause" if state["playing"] else "Play")
    play_button.on_clicked(lambda _event: set_playing(not state["playing"]))

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.text(
        0.55, 0.03,
        "Space: play/pause     ← / → : previous / next iteration",
        ha="center", fontsize=9,
    )

    timer = fig.canvas.new_timer(interval=interval)
    timer.add_callback(on_timer)
    timer.start()
    fig._simplex_controls = (timer, play_button, state)

    draw_current()

    if show:
        plt.show()

    return fig._simplex_controls


if __name__ == "__main__":
    import simplex as lp

    history = lp.simplex(
        lp.A,
        lp.b,
        lp.c,
        lp.basic_index,
        process=lp.process,
        variable_names=lp.variable_names,
    )
    visualize_simplex(history, lp.A, lp.variable_names, b=lp.b, c=lp.c)
