#!/usr/bin/env python3
"""Create deterministic benchmark plots from CSV artifacts.

Each trajectory gets separate timing figures. Timing is first averaged at each
sampling index across the independent repetitions. Summaries show the mean of
that averaged trace and its maximum; breakdowns use the same two statistics.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.facecolor": "#fbfbfb",
    "figure.facecolor": "white",
    "axes.edgecolor": "#555555",
    "grid.color": "#b8b8b8",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Embed TrueType fonts in vector output instead of emitting Type 3 glyphs.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# Colorblind-safe colors are reinforced with independent line styles and
# markers. The redundant encoding keeps curves legible in grayscale and for
# readers with color-vision deficiencies.
SOLVER_STYLES = {
    "daocp": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "hpipm": {"color": "#E69F00", "linestyle": "--", "marker": "s"},
    "daqp": {"color": "#009E73", "linestyle": "-.", "marker": "^"},
    "qpoases": {"color": "#CC79A7", "linestyle": ":", "marker": "D"},
}

SOLVER_ORDER = tuple(SOLVER_STYLES)
TRAJECTORY_ORDER = ("challenging", "daocp-worst")
BREAKDOWN_COMPONENTS = (
    ("condensing and expansion", "#8ECAE6", None),
    ("QP solver call and interface", "#D55E00", None),
)
SUMMARY_STYLES = (
    ("average", "#56B4E9", None),
    ("maximum sampling average", "#D55E00", None),
)
ROTOR_STYLES = (
    {"color": "#0072B2", "linestyle": "-"},
    {"color": "#E69F00", "linestyle": "--"},
    {"color": "#009E73", "linestyle": "-."},
    {"color": "#CC79A7", "linestyle": ":"},
)


def solver_style(solver, with_markers=True):
    style = dict(SOLVER_STYLES[solver])
    if with_markers:
        style.update(markevery=0.08, markersize=3.5, markerfacecolor="white",
                     markeredgewidth=0.8)
    else:
        style.pop("marker")
    return style


def load_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig, output, name):
    """Save a raster preview and a vector PDF with embedded fonts."""
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")


def ordered_trajectories(rows):
    available = {row["trajectory"] for row in rows}
    return [trajectory for trajectory in TRAJECTORY_ORDER if trajectory in available]


def ordered_solvers(rows):
    available = {row["solver"] for row in rows}
    return [solver for solver in SOLVER_ORDER if solver in available]


def summary_rows(summary, trajectory):
    rows = [row for row in summary if row["trajectory"] == trajectory]
    order = {solver: index for index, solver in enumerate(SOLVER_ORDER)}
    return sorted(rows, key=lambda row: order.get(row["solver"], len(order)))


def save_total_call_times_table(summary, output):
    """Write the final acados time_tot comparison as a readable text table."""
    trajectories = ordered_trajectories(summary)
    if any(row["trajectory"] == "overall" for row in summary):
        trajectories.append("overall")

    rows = [row for trajectory in trajectories
            for row in summary_rows(summary, trajectory)]
    headers = ("trajectory", "solver", "average [us]", "max avg [us]")
    formatted = [
        (
            row["trajectory"],
            row["solver"],
            f'{float(row["average_time_tot_us"]):.2f}',
            f'{float(row["max_per_sampling_average_time_tot_us"]):.2f}',
        )
        for row in rows
    ]
    widths = [max(len(header), *(len(row[index]) for row in formatted))
              for index, header in enumerate(headers)]

    def table_row(values):
        return "  ".join(
            value.ljust(widths[index]) if index < 2 else value.rjust(widths[index])
            for index, value in enumerate(values)
        )

    with (output / "total_acados_call_times.txt").open("w") as handle:
        handle.write("ACADOS total call times (time_tot)\n")
        handle.write("Average and maximum of the per-sampling averages, in microseconds.\n\n")
        handle.write(table_row(headers) + "\n")
        handle.write(table_row(tuple("-" * width for width in widths)) + "\n")
        for row in formatted:
            handle.write(table_row(row) + "\n")


def save_timing_summaries(summary, output):
    for trajectory in ordered_trajectories(summary):
        rows = summary_rows(summary, trajectory)
        labels = [row["solver"] for row in rows]
        average = np.array([float(row["average_time_tot_us"]) for row in rows])
        maximum = np.array([float(row["max_per_sampling_average_time_tot_us"]) for row in rows])
        x = np.arange(len(rows))
        width = 0.38
        fig, axis = plt.subplots(figsize=(8, 4.8))
        average_label, average_color, average_hatch = SUMMARY_STYLES[0]
        maximum_label, maximum_color, maximum_hatch = SUMMARY_STYLES[1]
        axis.bar(x-width/2, average, width, label=average_label,
                 color=average_color, edgecolor="#333333", linewidth=0.7,
                 hatch=average_hatch)
        axis.bar(x+width/2, maximum, width, label=maximum_label,
                 color=maximum_color, edgecolor="#333333", linewidth=0.7,
                 hatch=maximum_hatch)
        axis.set_title(f"{trajectory.capitalize()} trajectory")
        axis.set_ylabel("acados total solve time [µs]")
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
        fig.tight_layout()
        save_figure(fig, output, f"solve_time_summary_{trajectory}")
        plt.close(fig)


TIMING_COLUMNS = (
    "time_tot_us", "time_lin_us", "time_qp_us",
    "time_qp_solver_call_us", "time_qp_xcond_us",
)


def per_sampling_average_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["step"])].append(row)
    result = []
    for step, samples in sorted(grouped.items()):
        averaged = {column: float(np.mean([float(row[column]) for row in samples]))
                    for column in TIMING_COLUMNS}
        averaged["step"] = step
        averaged["time_s"] = float(np.mean([float(row["time_s"]) for row in samples]))
        result.append(averaged)
    return result


def save_timing_traces(steps, summary, output):
    summary_map = {(r["solver"], r["trajectory"]): r for r in summary}
    for trajectory in ordered_trajectories(steps):
        fig, axis = plt.subplots(figsize=(10, 4.8))
        groups = defaultdict(list)
        for row in steps:
            if row["trajectory"] == trajectory:
                groups[row["solver"]].append(row)
        for solver in ordered_solvers(steps):
            rows = groups.get(solver, [])
            if not rows:
                continue
            averaged_rows = per_sampling_average_rows(rows)
            time = [r["time_s"] for r in averaged_rows]
            solve = [r["time_tot_us"] for r in averaged_rows]
            style = solver_style(solver)
            axis.plot(time, solve, lw=0.9, alpha=0.85, label=solver, **style)
            stat = summary_map.get((solver, trajectory))
            if stat:
                color = SOLVER_STYLES[solver]["color"]
                axis.axhline(float(stat["average_time_tot_us"]), color=color, ls="--", lw=0.9)
                axis.axhline(float(stat["max_per_sampling_average_time_tot_us"]), color=color, ls=":", lw=0.9)
        axis.set_title(f"{trajectory.capitalize()} trajectory")
        axis.set_ylabel("time_tot [µs]")
        axis.set_xlabel("simulation time [s]")
        axis.grid(alpha=0.2)
        handles, labels = axis.get_legend_handles_labels()
        handles.extend((
            Line2D([0], [0], color="black", ls="--", lw=1.0, label="average"),
            Line2D([0], [0], color="black", ls=":", lw=1.0, label="maximum sampling average"),
        ))
        labels.extend(("average", "maximum sampling average"))
        axis.legend(handles, labels, ncol=3, frameon=False)
        fig.tight_layout()
        save_figure(fig, output, f"solve_time_traces_{trajectory}")
        plt.close(fig)


def timing_breakdown(row):
    qp_total = float(row["time_qp_us"])
    qp_solver = float(row["time_qp_solver_call_us"])
    condensing = float(row["time_qp_xcond_us"])
    qp_interface = max(0.0, qp_total - qp_solver - condensing)
    return np.array((condensing, qp_solver + qp_interface))


def save_timing_breakdowns(steps, output):
    for trajectory in ordered_trajectories(steps):
        trajectory_rows = [row for row in steps if row["trajectory"] == trajectory]
        solvers = ordered_solvers(trajectory_rows)
        rows_by_solver = {
            solver: per_sampling_average_rows(
                [row for row in trajectory_rows if row["solver"] == solver])
            for solver in solvers
        }
        # A 3.5-inch width preserves the configured font sizes when the PDF is
        # included at a single-column width in a typical two-column paper.
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 6.5))
        for axis, statistic, title in zip(
                axes, ("average", "maximum"),
                ("Average", "Worst case")):
            if statistic == "average":
                values = np.vstack([
                    np.mean([timing_breakdown(row) for row in rows_by_solver[solver]], axis=0)
                    for solver in solvers
                ])
            else:
                values = np.vstack([
                    max((timing_breakdown(row) for row in rows_by_solver[solver]), key=np.sum)
                    for solver in solvers
                ])
            bottoms = np.zeros(len(solvers))
            for component_index, (label, color, hatch) in enumerate(BREAKDOWN_COMPONENTS):
                component = values[:, component_index]
                axis.bar(solvers, component, bottom=bottoms, label=label,
                         color=color, edgecolor="black", linewidth=0.6, hatch=hatch)
                bottoms += component
            for index, total in enumerate(bottoms):
                axis.annotate(f"{total:.0f}", (index, total), xytext=(0, 3),
                              textcoords="offset points", ha="center", fontsize=10)
            axis.set_ylim(0.0, 1.12*float(np.max(bottoms)))
            axis.set_title(title)
            axis.set_ylabel(r"QP timing [$\mu$s]")
            axis.grid(axis="y", alpha=0.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.suptitle("Solver timing breakdown")
        fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False)
        fig.tight_layout(rect=(0, 0.13, 1, 0.96))
        save_figure(fig, output, f"solve_time_breakdown_{trajectory}")
        plt.close(fig)


def first_runs(steps):
    grouped = defaultdict(list)
    for row in steps:
        if row["repetition"] == "0":
            grouped[(row["solver"], row["trajectory"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["step"]))
    return grouped


def save_trajectory(groups, output):
    trajectories = sorted({key[1] for key in groups})
    fig, axes = plt.subplots(1, len(trajectories), figsize=(6*len(trajectories), 5), squeeze=False)
    for axis, trajectory in zip(axes[0], trajectories):
        reference_drawn = False
        for (solver, traj), rows in groups.items():
            if traj != trajectory:
                continue
            axis.plot([float(r["px"]) for r in rows], [float(r["py"]) for r in rows],
                      lw=1.1, label=solver, **solver_style(solver))
            if not reference_drawn:
                axis.plot([float(r["ref_px"]) for r in rows], [float(r["ref_py"]) for r in rows],
                          color="black", ls="--", lw=1.0, label="reference")
                reference_drawn = True
        axis.set_title(trajectory)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.axis("equal")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output, "trajectory")
    plt.close(fig)


def save_tracking(groups, output):
    trajectories = sorted({key[1] for key in groups})
    fig, axes = plt.subplots(len(trajectories), 1, figsize=(10, 3.4*len(trajectories)), squeeze=False)
    for axis, trajectory in zip(axes[:, 0], trajectories):
        for (solver, traj), rows in groups.items():
            if traj == trajectory:
                axis.plot([float(r["time_s"]) for r in rows], [float(r["position_error"]) for r in rows],
                          lw=1.0, label=solver, **solver_style(solver))
        axis.set_title(trajectory)
        axis.set_ylabel("position error [m]")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[-1, 0].set_xlabel("simulation time [s]")
    fig.tight_layout()
    save_figure(fig, output, "tracking_error")
    plt.close(fig)


def save_actuators(groups, output):
    trajectories = sorted({key[1] for key in groups})
    solvers = sorted({key[0] for key in groups})
    fig, axes = plt.subplots(len(trajectories), len(solvers),
                             figsize=(4.2*len(solvers), 3.1*len(trajectories)), squeeze=False)
    for row_idx, trajectory in enumerate(trajectories):
        for col_idx, solver in enumerate(solvers):
            axis = axes[row_idx, col_idx]
            rows = groups.get((solver, trajectory), [])
            time = [float(r["time_s"]) for r in rows]
            for rotor in range(4):
                axis.plot(time, [float(r[f"u{rotor}"]) for r in rows],
                          lw=0.9, label=f"u{rotor+1}", **ROTOR_STYLES[rotor])
            axis.axhline(0.0, color="black", lw=0.6)
            axis.axhline(8.5, color="black", lw=0.6)
            axis.set_ylim(-0.2, 8.8)
            axis.set_title(f"{solver} / {trajectory}")
            axis.set_ylabel("rotor thrust [N]")
            axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("simulation time [s]")
    axes[0, -1].legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, output, "actuator_usage")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    output = args.input.resolve()
    steps = load_csv(output / "steps.csv")
    summary = load_csv(output / "summary.csv")
    if not steps or not summary:
        raise SystemExit("no benchmark samples found")
    groups = first_runs(steps)
    save_total_call_times_table(summary, output)
    save_timing_summaries(summary, output)
    save_timing_traces(steps, summary, output)
    save_timing_breakdowns(steps, output)
    save_trajectory(groups, output)
    save_tracking(groups, output)
    save_actuators(groups, output)


if __name__ == "__main__":
    main()
